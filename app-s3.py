#!/usr/bin/env python3
# s3_ingest.py

import os
import sys
import json
import asyncio
import signal
from datetime import datetime
from dateutil import parser

import boto3
from botocore.config import Config

from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
from shared.common_code import (
    parse_last_sync_time,
    remove_missing_files,
    finalize_data_source,
    generate_md5_hash,
    handle_xlsx_blob_async,
    process_file_with_tika,
    shutdown_handler,
    get_event_loop,
    CENTRAL_TZ,
    UPLOADS_FOLDER
)

def get_s3_client(access_key, secret_key, region):
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    config = Config(retries={"max_attempts": 10, "mode": "standard"})
    return session.client("s3", region_name=region, config=config)

def list_s3_objects(s3_client, bucket_name, prefix=""):
    all_items = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket_name, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        resp = s3_client.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])
        for obj in contents:
            all_items.append(obj)

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    return all_items

def download_s3_object(s3_client, bucket_name, key, local_path):
    s3_client.download_file(bucket_name, key, local_path)

def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"S3 ingest event_data: {event_data}")
    except Exception:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            return ("No DataSource found", 404)
        if ds["sourceType"] != "s3":
            return (f"DataSource {data_source_id} is not s3.", 400)

        access_key = ds.get("s3AccessKey")
        secret_key = ds.get("s3SecretKey")
        region = ds.get("s3BucketRegion")
        bucket_name = ds.get("s3BucketName")
        folder_prefix = ds.get("s3FolderPath") or ""

        if not (access_key and secret_key and bucket_name):
            log.error("Missing S3 credentials or bucket in ds record.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Missing S3 credentials/bucket", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = parse_last_sync_time(last_sync_time)

        s3_client = get_s3_client(access_key, secret_key, region)
        s3_items = list_s3_objects(s3_client, bucket_name, folder_prefix)
        log.info(f"Found {len(s3_items)} objects in s3://{bucket_name}/{folder_prefix}")

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
        s3_file_keys = set()
        s3key_to_md5 = {}

        for obj in s3_items:
            s3key = obj["Key"]
            last_modified = obj["LastModified"]
            if last_modified.tzinfo is None:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, s3key)
            s3_file_keys.add(file_key)
            s3key_to_md5[s3key] = file_key

            if file_key not in db_file_keys:
                new_files.append((s3key, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((s3key, last_modified))
                elif not last_sync_dt:
                    updated_files.append((s3key, last_modified))

        remove_missing_files(db_file_keys, s3_file_keys, data_source_id, project_id, namespace=project_id)
        to_process = new_files + updated_files
        if not to_process:
            finalize_data_source(data_source_id, ds, new_status="processed")
            return ("No new or updated files", 200)

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)
        xlsx_tasks = []

        for s3key, last_modified in to_process:
            file_key = s3key_to_md5[s3key]
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(s3key)

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
                    source_type="s3",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            extension = "pdf"
            if "." in s3key:
                extension = s3key.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")

            try:
                download_s3_object(s3_client, bucket_name, s3key, local_tmp_path)
            except Exception as e:
                log.exception(f"Failed to download s3://{bucket_name}/{s3key}")
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
                source_type="s3",
                sub_id=sub_id
            )

            if final_status == "failed":
                log.error(f"S3 file {base_name} failed: {error_msg}")

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

            for (f_key, b_name, _, lm), result in zip(xlsx_tasks, xlsx_results):
                if isinstance(result, Exception):
                    log.error(f"XLSX error for {b_name}: {str(result)}")
                    psql.update_file_by_id(f_key, status="failed")
                    continue
                final_status, usage_credits, error_msg = result
                if final_status != "processed":
                    log.error(f"XLSX processing failed for {b_name}: {error_msg}")
                    psql.update_file_by_id(f_key, status="failed")
                else:
                    log.info(f"XLSX file {b_name} processed successfully.")
                    psql.update_file_by_id(f_key, status="processed")

        finalize_data_source(data_source_id, ds, new_status="processed")
        return ("OK", 200)

    except Exception as e:
        log.exception("Error in S3 ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            log.info(f"Created UPLOADS_FOLDER at {UPLOADS_FOLDER}")
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
