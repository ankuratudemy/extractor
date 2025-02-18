#!/usr/bin/env python3
# app-indexer.py

import os
import io
import json
import sys
import signal
from datetime import datetime

from flask import Flask, request
from google.cloud import storage
from werkzeug.utils import secure_filename

from shared.logging_config import log
from shared import psql, google_auth, security, google_pub_sub, file_processor
from shared.common_code import (
    process_file_with_tika,
    handle_xlsx_blob_async,
    generate_md5_hash,
    get_event_loop,
    shutdown_handler,
    CENTRAL_TZ,
    UPLOADS_FOLDER,
)

app = Flask(__name__)
app.debug = True

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
SERVER_URL = os.environ.get("SERVER_URL")

@app.route("/", methods=["POST"])
def event_handler():
    file_id = None
    try:
        event_data = request.get_json()
        log.info(f"Event Data: {event_data}")
        if not event_data:
            return ("No event data", 400)

        bucket_name = event_data.get("bucket")
        file_name = event_data.get("name")
        if not bucket_name or not file_name:
            return ("Missing bucket or file name in event data", 400)

        folder_name, file_name_only = os.path.split(file_name)
        filename = file_name_only
        temp_file_path = os.path.join(UPLOADS_FOLDER, filename)

        folder_name_parts = folder_name.split("/")
        if len(folder_name_parts) < 3:
            log.error("Invalid folder path format for GCS object.")
            return ("Invalid folder path format", 400)

        subscription_id = folder_name_parts[0]
        project_id = folder_name_parts[1]
        user_id = folder_name_parts[2]
        log.info(f"Subscription: {subscription_id}, Project: {project_id}, User: {user_id}")

        remaining_credits = psql.get_remaining_credits(subscription_id)
        log.info(f"Remaining credits for subscription {subscription_id}: {remaining_credits}")

        file_id = generate_md5_hash(subscription_id, project_id, filename)
        file_details = psql.fetch_file_by_id(file_id=file_id)
        if not file_details:
            log.error("No file record found in DB for this file_id.")
            return ("File record not found in DB", 400)

        if remaining_credits <= 0:
            log.error(f"No credits left for subscription {subscription_id}.")
            psql.update_file_status(file_id, status="no credits", page_count=0, updated_at=datetime.now())
            return ("No credits left for subscription", 402)

        # Download from GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(temp_file_path)
        log.info(f"Downloaded {file_name} to {temp_file_path}")

        psql.update_file_status(file_id, status="processing", page_count=0, updated_at=datetime.now())

        _, ext = os.path.splitext(filename)
        file_extension = ext[1:].lower() if ext else "pdf"

        # If it's XLSX or variants, handle the XLSX async approach
        if file_extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
            # In this code example, let's do it directly (synchronously or asynchronously).
            # For brevity, do synchronous approach calling your existing process_xlsx_blob:

            loop = get_event_loop()
            async def do_xlsx():
                return await handle_xlsx_blob_async(
                    file_id,
                    filename,
                    temp_file_path,
                    project_id,
                    data_source_id=None,
                    sub_for_hash=(subscription_id or "no_subscription"),
                    sub_id=subscription_id
                )

            result = loop.run_until_complete(do_xlsx())
            final_status, usage_credits, error_msg = result

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

            if final_status == "processed":
                psql.update_file_status(
                    file_id,
                    status="processed",
                    page_count=0,  # update if your XLSX logic returns a page count
                    updated_at=datetime.now(),
                )
                if usage_credits > 0:
                    message = json.dumps({
                        "subscription_id": subscription_id,
                        "user_id": user_id,
                        "project_id": project_id,
                        "creditsUsed": usage_credits,
                    })
                    google_pub_sub.publish_messages_with_retry_settings(
                        GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                    )
                return ("XLSX processed", 200)
            else:
                psql.update_file_status(file_id, status="failed", page_count=0, updated_at=datetime.now())
                return (f"XLSX processing failed: {error_msg}", 400)

        # Otherwise process with Tika flow
        final_status, page_count, error_msg = process_file_with_tika(
            file_key=file_id,
            base_name=filename,
            local_tmp_path=temp_file_path,
            extension=file_extension,
            project_id=project_id,
            data_source_id=None,  # This is an uploaded file, not a data source
            last_modified=datetime.now(CENTRAL_TZ),
            source_type="uploaded",
            sub_id=subscription_id
        )

        if final_status == "processed":
            # Build a JSON response if desired, or just return success
            return json.dumps({"status": "processed", "pageCount": page_count}), 200
        else:
            return (f"Processing failed: {error_msg}", 500)

    except Exception as e:
        log.exception(f"Failed processing file. Error: {str(e)}")
        if file_id:
            psql.update_file_status(file_id, status="failed", page_count=0, updated_at=datetime.now())
        return (f"Exception in event_handler: {e}", 500)


if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER, exist_ok=True)

    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
