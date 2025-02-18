#!/usr/bin/env python3
# app-indexer.py

import os
import io
import json
import sys
import ssl
import signal
import hashlib
from datetime import datetime

from flask import Flask, request
from google.cloud import storage
from werkzeug.utils import secure_filename

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_auth, security, google_pub_sub, file_processor

# --- Import all relevant helpers from your common_code file ---
from shared.common_code import (
    CENTRAL_TZ,
    SERVER_URL,
    get_event_loop,
    generate_md5_hash,
    process_pages_async,  # sends pages to Tika + does embedding
    process_xlsx_blob,  # for XLSX ingestion
    convert_to_pdf,  # for doc/docx/ppt/etc. -> PDF
    shutdown_handler,
)

app = Flask(__name__)
app.debug = True

# ----------------------------------------------------------------------------
# ENV VARIABLES
# ----------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER", "/tmp/uploads")  # default

app.config["UPLOAD_FOLDER"] = UPLOADS_FOLDER

# ----------------------------------------------------------------------------
# Initialize any clients or config as needed
# ----------------------------------------------------------------------------
if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------------
# Pinecone references are handled inside common_code, so we don't do it here.
# ----------------------------------------------------------------------------


def download_file(bucket_name: str, filename: str, temp_file_path: str) -> str:
    """
    Download file from GCS bucket to local temp_file_path.
    Returns content_type of the blob.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    blob.download_to_filename(temp_file_path)
    log.info(f"Downloaded {filename} from bucket {bucket_name} to {temp_file_path}")
    return blob.content_type


@app.route("/", methods=["POST"])
def event_handler():
    """
    Triggered by a GCS upload event. Processes a single file at a time:
    1) Check credits
    2) Download from GCS
    3) Depending on file extension, either:
       - Convert to PDF (if doc/docx/ppt, etc.) and parse PDF pages
       - Directly parse PDF pages
       - If XLSX, pass to process_xlsx_blob (like the S3 code)
       - If images, pass the image bytes as a single "page"
    4) Extract text with Tika + run embeddings (via process_pages_async)
    5) Update DB and publish credit usage
    """
    file_id = None
    try:
        event_data = request.get_json()
        log.info(f"Event Data: {event_data}")
        if not event_data:
            return ("No event data", 400)

        try:
            bucket_name = event_data.get("bucket")
            file_name = event_data.get("name")
        except (KeyError, AttributeError):
            log.error("Invalid bucket or file name in event data.")
            return ("Missing required fields in event data", 400)

        if not bucket_name or not file_name:
            return ("Missing bucket or file name in event data", 400)

        # Extract parts: (folder_name may look like subscriptionId/projectId/userId)
        folder_name, file_name_only = os.path.split(file_name)
        filename = file_name_only
        temp_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        # Attempt to parse subscriptionId/projectId/userId from path
        folder_name_parts = folder_name.split("/")
        if len(folder_name_parts) < 3:
            log.error("Invalid folder path format for GCS object.")
            return ("Invalid folder path format", 400)

        subscription_id = folder_name_parts[0]
        project_id = folder_name_parts[1]
        user_id = folder_name_parts[2]
        log.info(
            f"Subscription ID: {subscription_id}, Project ID: {project_id}, "
            f"User ID: {user_id}"
        )

        # Check remaining credits
        remaining_credits = psql.get_remaining_credits(subscription_id)
        log.info(
            f"Remaining credits for subscription {subscription_id}: {remaining_credits}"
        )

        # Generate a DB file_id
        file_id = generate_md5_hash(subscription_id, project_id, filename)
        file_details = psql.fetch_file_by_id(file_id=file_id)
        if not file_details:
            log.error("No file record found in DB for this file_id. Aborting.")
            return ("File record not found in DB", 400)

        if remaining_credits <= 0:
            log.error(f"No credits left for subscription {subscription_id}. Skipping.")
            psql.update_file_status(
                file_id=file_id,
                status="no credits",
                page_count=0,
                updated_at=datetime.now(),
            )
            return ("No credits left for subscription", 402)

        # We do have credits, proceed...
        content_type_header = download_file(bucket_name, file_name, temp_file_path)
        log.info(f"Generated file_id: {file_id}")
        psql.update_file_status(
            file_id=file_id,
            status="processing",
            page_count=0,
            updated_at=datetime.now(),
        )

        # Decide file extension
        # If the content type is not recognized, we fallback to extension from the file
        oFileExtMap = {
            "application/octet-stream": "use_extension",  # use file extension if content type is octet-stream
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/pdf": "pdf",
            "application/vnd.oasis.opendocument.text": "odt",
            "application/vnd.oasis.opendocument.spreadsheet": "ods",
            "application/vnd.oasis.opendocument.presentation": "odp",
            "application/vnd.oasis.opendocument.graphics": "odg",
            "application/vnd.oasis.opendocument.formula": "odf",
            "application/vnd.oasis.opendocument.flat.text": "fodt",
            "application/vnd.oasis.opendocument.flat.presentation": "fodp",
            "application/vnd.oasis.opendocument.flat.graphics": "fodg",
            "application/vnd.oasis.opendocument.spreadsheet-template": "ots",
            "application/vnd.oasis.opendocument.flat.spreadsheet-template": "fots",
            "application/vnd.lotus-1-2-3": "123",
            "application/dbase": "dbf",
            "text/html": "html",
            "application/vnd.lotus-screencam": "scm",
            "text/csv": "csv",
            "application/vnd.ms-excel": "xls",
            "application/vnd.ms-excel.template.macroenabled.12": "xltm",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template": "dotx",
            "application/vnd.ms-word.document.macroenabled.12": "docm",
            "application/vnd.ms-word.template.macroenabled.12": "dotm",
            "application/xml": "xml",
            "application/msword": "doc",
            "application/vnd.ms-word.document.macroenabled.12": "docm",
            "application/vnd.ms-word.template.macroenabled.12": "dotm",
            "application/rtf": "rtf",
            "application/wordperfect": "wpd",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.template": "xltx",
            "application/vnd.ms-excel.sheet.macroenabled.12": "xlsm",
            "application/vnd.ms-excel.template.macroenabled.12": "xltm",
            "application/vnd.corelqpw": "qpw",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.slideshow": "ppsx",
            "application/vnd.openxmlformats-officedocument.presentationml.slide": "ppmx",
            "application/vnd.openxmlformats-officedocument.presentationml.template": "potx",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.ms-powerpoint.slideshow.macroenabled.12": "ppsm",
            "application/vnd.ms-powerpoint.presentation.macroenabled.12": "pptm",
            "application/vnd.ms-powerpoint.addin.macroenabled.12": "ppam",
            "application/vnd.ms-powerpoint.slideshow.macroenabled.12": "ppsm",
            "application/vnd.ms-powerpoint.presentation.macroenabled.12": "pptm",
            "application/vnd.ms-powerpoint.addin.macroenabled.12": "ppam",
            "application/vnd.ms-powerpoint": "ppt",
            "application/vnd.ms-powerpoint.slideshow": "pps",
            "application/vnd.ms-powerpoint.presentation": "ppt",
            "application/vnd.ms-powerpoint.addin": "ppa",
            # Email formats
            "message/rfc822": "eml",  # EML format
            "application/vnd.ms-outlook": "msg",  # MSG format
            "application/mbox": "mbox",  # MBOX format
            "application/vnd.ms-outlook": "pst",  # PST format
            "application/ost": "ost",  # OST format
            "application/emlx": "emlx",  # EMLX format
            "application/dbx": "dbx",  # DBX format
            "application/dat": "dat",  # Windows Mail (.dat) format
            # Image formats
            "image/jpeg": "jpg",  # JPEG format
            "image/png": "png",  # PNG format
            "image/gif": "gif",  # GIF format
            "image/tiff": "tiff",  # TIFF format
            "image/bmp": "bmp",  # BMP format
        }
        if content_type_header in oFileExtMap:
            file_extension = oFileExtMap[content_type_header]
            if file_extension == "use_extension":
                _, ext = os.path.splitext(filename)
                file_extension = ext[1:].lower() if ext else ""
        else:
            # If we can't map content type, fallback to extension
            _, ext = os.path.splitext(filename)
            file_extension = ext[1:].lower()

        log.info(f"Resolved extension for {filename} => '{file_extension}'")

        # Prepare Tika headers (Bearer token)
        bearer_token = google_auth.impersonated_id_token(SERVER_URL).json()["token"]
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        processed_pages = 0

        # ----------------------------------------
        # XLSX Flow (like S3 code)
        # ----------------------------------------
        if file_extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
            # We call process_xlsx_blob from common_code, passing data_source_id=None,
            # "no_subscription" (or subscription_id) for sub_for_hash, etc.
            # Because app-indexer handles only one file at a time, we do a direct call:
            try:
                sub_for_hash = subscription_id if subscription_id else "no_subscription"
                # local_tmp_path is temp_file_path
                status, usage_credits, error_msg = loop.run_until_complete(
                    process_xlsx_blob(
                        file_id,
                        filename,
                        temp_file_path,
                        project_id,
                        data_source_id=None,  # We treat this as an uploaded file
                        sub_for_hash=sub_for_hash,
                        sub_id=subscription_id,
                    )
                )
                # Remove local file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                if status == "processed":
                    processed_pages = 0  # If your XLSX server returns chunk/page count, set it properly
                    psql.update_file_status(
                        file_id=file_id,
                        status="processed",
                        page_count=processed_pages,
                        updated_at=datetime.now(),
                    )
                    if usage_credits > 0:
                        # Publish usage
                        message = json.dumps(
                            {
                                "subscription_id": subscription_id,
                                "user_id": user_id,
                                "project_id": project_id,
                                "creditsUsed": usage_credits,
                            }
                        )
                        google_pub_sub.publish_messages_with_retry_settings(
                            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                        )
                    # Return success
                    return (
                        json.dumps({"status": "processed"}),
                        200,
                        {"Content-Type": "application/json; charset=utf-8"},
                    )
                else:
                    psql.update_file_status(
                        file_id=file_id,
                        status="failed",
                        page_count=0,
                        updated_at=datetime.now(),
                    )
                    return (
                        f"XLSX processing failed: {error_msg}",
                        400,
                    )

            except Exception as e:
                log.exception(f"Failed processing XLSX: {str(e)}")
                psql.update_file_status(
                    file_id=file_id,
                    status="failed",
                    page_count=0,
                    updated_at=datetime.now(),
                )
                # Clean up local
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return (f"XLSX processing exception: {e}", 500)

        # ----------------------------------------
        # PDF or other convertible doc flow
        # ----------------------------------------
        pages = []
        num_pages = 0

        try:
            if file_extension == "pdf":
                with open(temp_file_path, "rb") as f:
                    pdf_data = f.read()
                pages, num_pages = file_processor.split_pdf(pdf_data)
                del pdf_data

            elif file_extension in [
                "docx",
                "odt",
                "odp",
                "odg",
                "odf",
                "fodt",
                "fodp",
                "fodg",
                "123",
                "dbf",
                "scm",
                "dotx",
                "docm",
                "dotm",
                "xml",
                "doc",
                "qpw",
                "pptx",
                "ppsx",
                "ppmx",
                "potx",
                "pptm",
                "ppam",
                "ppsm",
                "pptm",
                "ppam",
                "ppt",
                "pps",
                "ppt",
                "ppa",
                "rtf",
            ]:
                pdf_data = convert_to_pdf(temp_file_path, file_extension)
                if pdf_data:
                    pages, num_pages = file_processor.split_pdf(pdf_data)
                else:
                    log.error("Conversion to PDF failed.")
                    psql.update_file_status(
                        file_id=file_id,
                        status="failed",
                        page_count=0,
                        updated_at=datetime.now(),
                    )
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    return ("Conversion to PDF failed.", 400)
                del pdf_data

            elif file_extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
                # Treat as a single "page"
                with open(temp_file_path, "rb") as f:
                    image_data = f.read()
                pages = [("1", io.BytesIO(image_data))]
                num_pages = len(pages)
                del image_data

            # elif file_extension in ['csv', 'xls', 'xltm', 'xltx', 'xlsx', 'tsv', 'ots']:
            #     pages = file_processor.split_excel(uploaded_file.read())
            #     num_pages = len(pages)

            elif file_extension in [
                "eml",
                "msg",
                "pst",
                "ost",
                "mbox",
                "dbx",
                "dat",
                "emlx",
            ]:
                with open(temp_file_path, "rb") as f:
                    msg_data = f.read()
                pages = [("1", io.BytesIO(msg_data))]
                num_pages = len(pages)

            elif file_extension in ["ods"]:
                with open(temp_file_path, "rb") as f:
                    ods_data = f.read()
                pages = file_processor.split_ods(ods_data)
                num_pages = len(pages)

            else:
                # If needed, handle excel/ods/other here or return unsupported
                log.error(f"Unsupported file extension: {file_extension}")
                psql.update_file_status(
                    file_id=file_id,
                    status="failed",
                    page_count=0,
                    updated_at=datetime.now(),
                )
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                return ("Unsupported file format.", 400)

            # Remove the original local file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            log.exception(f"Error splitting/converting file {filename}: {e}")
            psql.update_file_status(
                file_id=file_id,
                status="failed",
                page_count=0,
                updated_at=datetime.now(),
            )
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return (f"Error splitting file: {e}", 500)

        # Finally, run Tika + embeddings
        try:
            # data_source_id=None, sourceType="uploaded"
            results = loop.run_until_complete(
                process_pages_async(
                    pages,
                    headers,
                    filename,
                    namespace=project_id,
                    file_id=file_id,
                    data_source_id=None,
                    last_modified=datetime.now(CENTRAL_TZ),
                    sourceType="uploaded",
                )
            )
        except Exception as e:
            log.exception(f"Failed Tika/embedding for {filename}: {e}")
            psql.update_file_status(
                file_id=file_id,
                status="failed",
                page_count=0,
                updated_at=datetime.now(),
            )
            return (f"Embedding or Tika extraction failed: {e}", 500)

        # results is list of tuples: [(text_content, page_num), ...]
        # Build a JSON response if desired
        json_output = []
        processed_pages = 0
        for text_content, page_num in results:
            clean_text = text_content.strip()
            if not clean_text:
                continue
            json_output.append({"page": page_num, "text": clean_text})
            processed_pages += 1

        # Suppose each processed page costs 1.5 credits
        credits_used = processed_pages * 1.5
        message = json.dumps(
            {
                "subscription_id": subscription_id,
                "user_id": user_id,
                "project_id": project_id,
                "creditsUsed": credits_used,
            }
        )
        log.info(f"Publishing credit usage: {message}")
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
        )

        # Update DB status
        psql.update_file_status(
            file_id=file_id,
            status="processed",
            page_count=processed_pages,
            updated_at=datetime.now(),
        )

        json_string = json.dumps(json_output, indent=4)
        return (json_string, 200, {"Content-Type": "application/json; charset=utf-8"})

    except Exception as e:
        # Catchall for unexpected errors
        log.exception(f"Failed processing file. Error: {str(e)}")
        if file_id:
            psql.update_file_status(
                file_id=file_id,
                status="failed",
                page_count=0,
                updated_at=datetime.now(),
            )
        return (f"Exception in event_handler: {e}", 500)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
