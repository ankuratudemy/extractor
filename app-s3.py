#!/usr/bin/env python3
# s3_ingest.py

import os
import io
import sys
import json
import signal
import asyncio
import aiohttp
import requests
from datetime import datetime
from dateutil import parser

import boto3
from botocore.config import Config

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor

# Import the shared helpers + constants
from shared.common_code import (
    ensure_timezone_aware,
    get_event_loop,
    generate_md5_hash,
    compute_next_sync_time,
    remove_file_from_db_and_pinecone,
    convert_to_pdf,
    process_pages_async,  # for PDF/docx/pptx flow
    process_xlsx_blob,  # if you need XLSX flow in S3
    CENTRAL_TZ,
    UPLOADS_FOLDER,
    SERVER_URL,
    SERVER_DOMAIN,
    GCP_PROJECT_ID,
    GCP_CREDIT_USAGE_TOPIC,
    shutdown_handler,
)


def get_s3_client(access_key, secret_key, region):
    """
    Returns a Boto3 S3 client with the provided credentials and region.
    """
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    config = Config(retries={"max_attempts": 10, "mode": "standard"})
    return session.client("s3", region_name=region, config=config)


def list_s3_objects(s3_client, bucket_name, prefix=""):
    """
    Lists all objects in the given bucket under `prefix` (if provided).
    """
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
    """
    Downloads the object from S3 to `local_path`.
    """
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
    except Exception as e:
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
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not s3.", 400)

        access_key = ds.get("s3AccessKey")
        secret_key = ds.get("s3SecretKey")
        region = ds.get("s3BucketRegion")
        bucket_name = ds.get("s3BucketName")
        folder_prefix = ds.get("s3FolderPath") or ""

        if not (access_key and secret_key and bucket_name):
            log.error("Missing S3 credentials or bucket in ds record.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Missing S3 credentials/bucket", 400)

        # last sync time
        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None

        s3_client = get_s3_client(access_key, secret_key, region)
        s3_items = list_s3_objects(s3_client, bucket_name, folder_prefix)
        log.info(
            f"Found {len(s3_items)} objects under s3://{bucket_name}/{folder_prefix}"
        )

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

        # categorize new/updated
        for obj in s3_items:
            s3key = obj["Key"]
            last_modified = obj["LastModified"]
            if not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            file_key = generate_md5_hash(
                sub_for_hash, project_id, data_source_id, s3key
            )
            s3_file_keys.add(file_key)
            s3key_to_md5[s3key] = file_key

            if file_key not in db_file_keys:
                new_files.append((s3key, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((s3key, last_modified))
                elif not last_sync_dt:
                    updated_files.append((s3key, last_modified))

        # detect removed
        removed_keys = db_file_keys - s3_file_keys
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(
                r_key, data_source_id, project_id, namespace=project_id
            )

        to_process = new_files + updated_files
        if not to_process:
            now_dt = datetime.now(CENTRAL_TZ)
            psql.update_data_source_by_id(
                data_source_id,
                status="processed",
                lastSyncTime=now_dt.isoformat(),
                nextSyncTime=compute_next_sync_time(
                    now_dt, ds.get("syncOption")
                ).isoformat(),
            )
            return ("No new or updated files", 200)

        if not SERVER_URL:
            log.error("SERVER_URL is not set, cannot do Tika processing.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("SERVER_URL missing", 500)

        try:
            bearer_token = google_auth.impersonated_id_token(
                serverurl=SERVER_DOMAIN
            ).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token:")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Failed to get impersonated ID token", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        # If you want to handle XLSX in S3, you can queue them like in azure:
        xlsx_tasks = []

        for s3key, last_modified in to_process:
            file_key = s3key_to_md5[s3key]
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(s3key)

            # Insert or update file in DB
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
                psql.update_file_by_id(
                    file_key, status="processing", updatedAt=now_dt.isoformat()
                )

            # Download locally
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

            # If XLSX, queue an async XLSX process (if you want S3 XLSX flow):
            if extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((file_key, base_name, local_tmp_path, last_modified))
                continue

            if extension in [
                "pdf",
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
                "jpg",
                "jpeg",
                "png",
                "gif",
                "tiff",
                "bmp",
                "eml",
                "msg",
                "pst",
                "ost",
                "mbox",
                "dbx",
                "dat",
                "emlx",
                "ods",
            ]:

                def process_local_file(temp_path, file_ext):
                    if file_ext == "pdf":
                        with open(temp_path, "rb") as f_in:
                            pdf_data = f_in.read()
                        return file_processor.split_pdf(pdf_data)
                    elif file_ext in [
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
                        pdf_data = convert_to_pdf(temp_path, file_ext)
                        if pdf_data:
                            return file_processor.split_pdf(pdf_data)
                        else:
                            raise ValueError("Conversion to PDF failed for Azure file.")
                    elif extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
                        # Treat as a single "page"
                        with open(temp_path, "rb") as f:
                            image_data = f.read()
                        pages = [("1", io.BytesIO(image_data))]
                        num_pages = len(pages)
                        return pages, num_pages
                    elif extension in [
                        "eml",
                        "msg",
                        "pst",
                        "ost",
                        "mbox",
                        "dbx",
                        "dat",
                        "emlx",
                    ]:
                        with open(temp_path, "rb") as f:
                            msg_data = f.read()
                        pages = [("1", io.BytesIO(msg_data))]
                        num_pages = len(pages)
                        return pages, num_pages
                    elif extension in ["ods"]:
                        with open(temp_path, "rb") as f:
                            ods_data = f.read()
                        pages = file_processor.split_ods(ods_data)
                        num_pages = len(pages)
                        return pages, num_pages
                    raise ValueError("Unsupported file format in Azure ingestion logic")

                try:
                    final_pages, final_num_pages = process_local_file(
                        local_tmp_path, extension
                    )
                except Exception as e:
                    log.error(f"Failed processing {base_name} as {extension}: {str(e)}")
                    psql.update_file_by_id(
                        file_key, status="failed", updatedAt=now_dt.isoformat()
                    )
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
                        sourceType="s3",
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding for {base_name}")
                psql.update_file_by_id(file_key, status="failed")
                continue

            # Mark processed
            psql.update_file_by_id(file_key, status="processed", pageCount=len(results))

            used_credits = len(results) * 1.5
            if sub_id:
                msg = json.dumps(
                    {
                        "subscription_id": sub_id,
                        "data_source_id": data_source_id,
                        "project_id": project_id,
                        "creditsUsed": used_credits,
                    }
                )
                try:
                    google_pub_sub.publish_messages_with_retry_settings(
                        GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=msg
                    )
                except Exception as e:
                    log.warning(f"Publish usage failed for S3 file {base_name}: {e}")

        # If handling XLSX in S3, do parallel tasks:
        if xlsx_tasks:
            log.info(f"Processing {len(xlsx_tasks)} XLSX files in parallel for S3.")

            async def process_all_xlsx_tasks(xlsx_files):
                tasks = []
                for f_key, b_name, tmp_path, lm in xlsx_files:
                    tasks.append(
                        process_xlsx_blob(
                            f_key,
                            b_name,
                            tmp_path,
                            project_id,
                            data_source_id,
                            sub_for_hash,
                            sub_id,
                        )
                    )
                return await asyncio.gather(*tasks, return_exceptions=True)

            xlsx_results = loop.run_until_complete(process_all_xlsx_tasks(xlsx_tasks))

            # xlsx_results => list of (final_status, usage_credits, error_msg)
            for (f_key, b_name, _, lm), (final_status, usage_credits, error_msg) in zip(
                xlsx_tasks, xlsx_results
            ):
                now_dt = datetime.now(CENTRAL_TZ)
                if final_status == "processed":
                    psql.update_file_by_id(
                        f_key, status="processed", updatedAt=now_dt.isoformat()
                    )
                    if usage_credits > 0 and sub_id:
                        msg = json.dumps(
                            {
                                "subscription_id": sub_id,
                                "data_source_id": data_source_id,
                                "project_id": project_id,
                                "creditsUsed": usage_credits,
                            }
                        )
                        try:
                            google_pub_sub.publish_messages_with_retry_settings(
                                GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=msg
                            )
                        except Exception as e:
                            log.warning(
                                f"Publish usage failed for XLSX file {b_name}: {e}"
                            )
                else:
                    log.error(f"XLSX processing failed for {b_name}. {error_msg}")
                    psql.update_file_by_id(
                        f_key, status="failed", updatedAt=now_dt.isoformat()
                    )

        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=compute_next_sync_time(
                final_now_dt, ds.get("syncOption")
            ).isoformat(),
        )
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

    if not SERVER_URL:
        log.error("SERVER_URL is not set.")
        sys.exit(1)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
