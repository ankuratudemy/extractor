#!/usr/bin/env python3
# gcp_ingest.py

import os
import sys
import json
import signal
import asyncio
from datetime import datetime
from google.cloud import storage

from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
from shared.common_code import (
    parse_last_sync_time,
    remove_missing_files,
    compute_next_sync_time,
    finalize_data_source,
    generate_md5_hash,
    handle_xlsx_blob_async,
    process_file_with_tika,
    shutdown_handler,
    get_event_loop,
    ensure_timezone_aware,
    CENTRAL_TZ,
    UPLOADS_FOLDER,
)

def connect_gcp_storage(credential_json):
    from google.oauth2 import service_account
    info = json.loads(credential_json)
    creds = service_account.Credentials.from_service_account_info(info)
    client = storage.Client(credentials=creds)
    log.info("Successfully connected to GCP Storage.")
    return client

def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"GCP ingest - parsed event_data: {event_data}")
    except Exception:
        log.error("Failed to parse DATA_SOURCE_CONFIG for GCP.")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    # Mark DS as 'processing'
    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            log.error(f"No DataSource found for id={data_source_id}.")
            return ("No DataSource found", 404)
        if ds["sourceType"] != "gcpBucket":
            return (f"DataSource {data_source_id} is not gcpBucket.", 400)

        credential_json = ds.get("gcpCredentialJson")
        bucket_name = ds.get("gcpBucketName")
        folder_prefix = ds.get("gcpFolderPath") or ""
        if not credential_json or not bucket_name:
            log.error("Missing GCP credentials or bucket in DS record.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Missing gcpCredentialJson or gcpBucketName", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = parse_last_sync_time(last_sync_time)

        # Connect
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
                continue
            gcp_key = blob.name
            last_modified = ensure_timezone_aware(blob.updated) if blob.updated else datetime.now(CENTRAL_TZ)

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            gcp_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((blob, gcp_key, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))
                elif not last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))

        remove_missing_files(db_file_keys, gcp_file_keys, data_source_id, project_id, namespace=project_id)
        to_process = new_files + updated_files
        log.info(f"New => {len(new_files)}, Updated => {len(updated_files)}")

        if not to_process:
            finalize_data_source(data_source_id, ds, new_status="processed")
            return ("No new/updated files", 200)

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        xlsx_tasks = []

        for blob, gcp_key, last_modified in to_process:
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(gcp_key)

            # Insert / update DB
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

            # Download
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

            if extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((file_key, base_name, local_tmp_path, last_modified))
                continue

            final_status, page_count, error_msg = process_file_with_tika(
                file_key=file_key,
                base_name=base_name,
                local_tmp_path=local_tmp_path,
                extension=extension,
                project_id=project_id,
                data_source_id=data_source_id,
                last_modified=last_modified,
                source_type="gcpBucket",
                sub_id=sub_id
            )

            if final_status == "failed":
                log.error(f"GCP file {base_name} failed: {error_msg}")

        # XLSX in parallel
        if xlsx_tasks:
            async def process_all_xlsx_tasks(xlsx_files):
                tasks = []
                for f_key, b_name, tmp_path, lm in xlsx_files:
                    tasks.append(
                        handle_xlsx_blob_async(
                            f_key,
                            b_name,
                            tmp_path,
                            project_id,
                            data_source_id,
                            sub_for_hash,
                            sub_id
                        )
                    )
                return await asyncio.gather(*tasks, return_exceptions=True)

            xlsx_results = loop.run_until_complete(process_all_xlsx_tasks(xlsx_tasks))
            for (f_key, b_name, tmp_path, lm), result in zip(xlsx_tasks, xlsx_results):
                if isinstance(result, Exception):
                    log.error(f"XLSX error for {b_name}: {str(result)}")
                    psql.update_file_by_id(f_key, status="failed")
                    continue
                final_status, usage_credits, error_msg = result
                if final_status != "processed":
                    log.error(f"XLSX failed for {b_name}: {error_msg}")
                    psql.update_file_by_id(f_key, status="failed")
                else:
                    log.info(f"XLSX file {b_name} processed successfully.")
                    psql.update_file_by_id(f_key, status="processed")

        finalize_data_source(data_source_id, ds, new_status="processed")
        return ("OK", 200)

    except Exception as e:
        log.error(f"Error in GCP ingestion flow: {e}")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
