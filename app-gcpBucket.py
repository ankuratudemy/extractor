#!/usr/bin/env python3
# gcp_ingest.py

import os
import sys
import json
import time
import signal
from datetime import datetime
from dateutil import parser
import asyncio
import aiohttp
from google.cloud import storage

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor

# Import your shared code from `common_code.py`
from shared.common_code import (
    CENTRAL_TZ,
    UPLOADS_FOLDER,
    SERVER_URL,
    SERVER_DOMAIN,
    GCP_PROJECT_ID,
    GCP_CREDIT_USAGE_TOPIC,
    ensure_timezone_aware,
    get_event_loop,
    generate_md5_hash,
    compute_next_sync_time,
    remove_file_from_db_and_pinecone,
    convert_to_pdf,
    process_pages_async,
    process_xlsx_blob,
    shutdown_handler,
)

def connect_gcp_storage(credential_json):
    """
    Connect to GCP Storage using the provided service account JSON.
    """
    from google.oauth2 import service_account
    try:
        info = json.loads(credential_json)
        creds = service_account.Credentials.from_service_account_info(info)
        client = storage.Client(credentials=creds)
        log.info("Successfully connected to GCP Storage.")
        return client
    except Exception as e:
        log.error("Failed to parse or use GCP credentials.")
        raise

def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(0)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"GCP ingest - parsed event_data: {event_data}")
    except Exception as e:
        log.error("Failed to parse DATA_SOURCE_CONFIG for GCP.")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    # Mark DataSource as 'processing'
    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            log.error(f"No DataSource found for id={data_source_id}.")
            return ("No DataSource found", 404)
        if ds["sourceType"] != "gcpBucket":
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not gcpBucket.", 400)

        credential_json = ds.get("gcpCredentialJson")
        bucket_name = ds.get("gcpBucketName")
        folder_prefix = ds.get("gcpFolderPath") or ""
        if not credential_json or not bucket_name:
            log.error("Missing GCP credentials or bucket in DataSource record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Missing gcpCredentialJson or gcpBucketName", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None

        # Connect to GCP Storage
        client = connect_gcp_storage(credential_json)
        bucket = client.bucket(bucket_name)
        all_blobs = list(bucket.list_blobs(prefix=folder_prefix))
        log.info(f"Found {len(all_blobs)} blobs in gs://{bucket_name}/{folder_prefix}")

        existing_files = psql.fetch_files_by_data_source_id(data_source_id)
        db_file_keys = set(ef["id"] for ef in existing_files)

        sub_id = None
        if project_id:
            proj_details = psql.get_project_details(project_id)
            if proj_details:
                sub_id = proj_details.get("subscriptionId")
        sub_for_hash = sub_id if sub_id else "no_subscription"

        new_files = []
        updated_files = []
        gcp_file_keys = set()

        for blob in all_blobs:
            if blob.name.endswith("/"):
                # skip folder placeholders
                continue
            gcp_key = blob.name
            last_modified = blob.updated
            if last_modified and not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)
            else:
                last_modified = last_modified or datetime.now(CENTRAL_TZ)

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            gcp_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((blob, gcp_key, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))
                elif not last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))

        removed_keys = db_file_keys - gcp_file_keys
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(r_key, data_source_id, project_id, project_id)

        log.info(f"New files => {len(new_files)}, Updated => {len(updated_files)}")
        to_process = new_files + updated_files
        if not to_process:
            now_dt = datetime.now(CENTRAL_TZ)
            psql.update_data_source_by_id(
                data_source_id,
                status="processed",
                lastSyncTime=now_dt.isoformat(),
                nextSyncTime=compute_next_sync_time(now_dt, ds.get("syncOption")).isoformat(),
            )
            return ("No new/updated files, done", 200)

        if not SERVER_URL:
            log.error("SERVER_URL is not set. Cannot proceed with Tika processing.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("No SERVER_URL", 500)

        try:
            bearer_token = google_auth.impersonated_id_token(serverurl=SERVER_DOMAIN).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token:")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Failed to obtain impersonated ID token.", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        # We'll store XLSX tasks for parallel processing
        xlsx_tasks = []

        for blob, gcp_key, last_modified in to_process:
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(gcp_key)

            # Insert / update DB record
            if file_key not in db_file_keys_list:
                psql.add_new_file(
                    file_id=file_key,
                    status="processing",
                    created_at=now_dt,
                    updated_at=now_dt,
                    name=base_name,
                    file_type="unknown",
                    file_key=f"{sub_for_hash}-{project_id}-{file_key}",
                    page_count=0,
                    upload_url="None",
                    source_type="gcpBucket",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            extension = "pdf"
            if "." in gcp_key:
                extension = gcp_key.rsplit(".", 1)[-1].lower()
            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")

            try:
                blob.download_to_filename(local_tmp_path)
            except Exception as e:
                log.error(f"Failed to download gs://{bucket_name}/{gcp_key}: {e}")
                psql.update_file_by_id(file_key, status="failed")
                continue

            if extension == "xlsx":
                # Queue XLSX tasks
                xlsx_tasks.append((file_key, base_name, local_tmp_path, last_modified))
                continue

            # PDF / docx / pptx logic
            def process_local_file(temp_path, file_ext):
                if file_ext == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_ext in ["docx", "pptx"]:
                    pdf_data = convert_to_pdf(temp_path, file_ext)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed.")
                raise ValueError(f"Unsupported file format: {file_ext}")

            try:
                final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
            except Exception as e:
                log.error(f"Failed processing {base_name} as {extension}: {e}")
                psql.update_file_by_id(file_key, status="failed")
                if os.path.exists(local_tmp_path):
                    os.remove(local_tmp_path)
                continue

            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

            try:
                results = loop.run_until_complete(
                    process_pages_async(
                        final_pages,
                        headers,
                        base_name,
                        project_id,
                        file_key,
                        data_source_id,
                        last_modified=last_modified,
                        sourceType="gcpBucket"
                    )
                )
            except Exception as e:
                log.error(f"Failed Tika/embedding for {base_name}: {e}")
                psql.update_file_by_id(file_key, status="failed")
                continue

            psql.update_file_by_id(
                file_key,
                status="processed",
                pageCount=len(results),
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )

            used_credits = len(results) * 1.5
            if sub_id:
                message = json.dumps({
                    "subscription_id": sub_id,
                    "data_source_id": data_source_id,
                    "project_id": project_id,
                    "creditsUsed": used_credits,
                })
                try:
                    google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message)
                except Exception as e:
                    log.warning(f"Publish usage failed for file {base_name}: {e}")

        # Handle XLSX tasks in parallel
        if xlsx_tasks:
            log.info(f"Processing {len(xlsx_tasks)} XLSX files in parallel for GCP.")
            async def process_all_xlsx_tasks(xlsx_files):
                tasks = []
                async with aiohttp.ClientSession() as session:
                    for (f_key, b_name, tmp_path, lm) in xlsx_files:
                        tasks.append(
                            process_xlsx_blob(
                                session,
                                f_key,
                                b_name,
                                tmp_path,
                                project_id,
                                data_source_id,
                                sub_for_hash,
                                sub_id
                            )
                        )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results

            xlsx_results = loop.run_until_complete(process_all_xlsx_tasks(xlsx_tasks))

            for ((f_key, b_name, _, lm), (final_status, usage_credits, error_msg)) in zip(xlsx_tasks, xlsx_results):
                now_dt = datetime.now(CENTRAL_TZ)
                if final_status == "processed":
                    psql.update_file_by_id(f_key, status="processed", updatedAt=now_dt.isoformat())
                    if usage_credits > 0 and sub_id:
                        msg = json.dumps({
                            "subscription_id": sub_id,
                            "data_source_id": data_source_id,
                            "project_id": project_id,
                            "creditsUsed": usage_credits,
                        })
                        try:
                            google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=msg)
                        except Exception as e:
                            log.warning(f"Publish usage failed for XLSX file {b_name}: {e}")
                else:
                    log.error(f"XLSX processing failed for {b_name}. {error_msg}")
                    psql.update_file_by_id(f_key, status="failed", updatedAt=now_dt.isoformat())

        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=compute_next_sync_time(final_now_dt, ds.get("syncOption")).isoformat(),
        )
        return ("OK", 200)

    except Exception as e:
        log.error(f"Error in GCP ingestion flow: {e}")
        psql.update_data_source_by_id(data_source_id, status="error")
        return (str(e), 500)

if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)

    if not SERVER_URL:
        log.error("SERVER_URL is not set correctly.")
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
