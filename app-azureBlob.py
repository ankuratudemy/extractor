#!/usr/bin/env python3
# azure_ingest.py

import os
import sys
import json
import io
import time
import signal
import ssl
import hashlib
import asyncio
import aiohttp
import tiktoken
import subprocess
import requests  # for PDF flow and docx flow, if needed
from datetime import datetime, timedelta
from typing import List, Tuple
from zoneinfo import ZoneInfo

from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
from shared.common_code import (
    ensure_timezone_aware,
    get_event_loop,
    generate_md5_hash,
    compute_next_sync_time,
    remove_file_from_db_and_pinecone,
    convert_to_pdf,
    process_pages_async,
    process_xlsx_blob,
    CENTRAL_TZ,
    UPLOADS_FOLDER,
    SERVER_URL,
    GCP_CREDIT_USAGE_TOPIC,
    GCP_PROJECT_ID,
    SERVER_DOMAIN,
)

# Azure blob
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError


# ----------------------------------------------------------------------------
# MAIN RUN
# ----------------------------------------------------------------------------
def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"Azure ingest - parsed event_data: {event_data}")
    except Exception as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG for Azure.")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        log.info(f"DataSource from db: {ds}")
        if not ds:
            return ("No DataSource found for id={}".format(data_source_id), 404)
        if ds["sourceType"] != "azureBlob":
            return (
                f"DataSource {data_source_id} is {ds['sourceType']}, not azureBlob.",
                400,
            )

        azure_account_name = ds.get("azureStorageAccountName")
        azure_account_key = ds.get("azureStorageAccountKey")
        container_name = ds.get("azureContainerName")
        folder_prefix = ds.get("azureFolderPath") or ""

        if not azure_account_name or not azure_account_key or not container_name:
            log.error("Missing Azure credentials or container in ds record.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Missing azure credentials or containerName", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None

        # Connect to Azure Blob
        conn_str = (
            f"DefaultEndpointsProtocol=https;AccountName={azure_account_name};"
            f"AccountKey={azure_account_key};EndpointSuffix=core.windows.net"
        )
        service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = service_client.get_container_client(container_name)

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
        azure_file_keys = set()

        # list blobs with prefix
        blob_list = container_client.list_blobs(name_starts_with=folder_prefix)
        all_blobs = list(blob_list)

        log.info(
            f"Found {len(all_blobs)} objects in container={container_name}, prefix={folder_prefix}"
        )

        for blob in all_blobs:
            if blob.name.endswith("/"):
                continue
            blob_name = blob.name
            last_modified = blob.last_modified
            if not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            file_key = generate_md5_hash(
                sub_for_hash, project_id, data_source_id, blob_name
            )
            azure_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((blob, blob_name, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((blob, blob_name, last_modified))
                elif not last_sync_dt:
                    updated_files.append((blob, blob_name, last_modified))

        removed_keys = db_file_keys - azure_file_keys
        log.info(f"Removed keys from DB: {removed_keys}")
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(
                r_key, data_source_id, project_id, namespace=project_id
            )

        log.info(f"New files => {len(new_files)}, Updated => {len(updated_files)}")
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
            return ("No new/updated files, done", 200)

        # Tika server token (for docx/pptx/pdfs)
        if not SERVER_URL:
            log.error("SERVER_URL is not set. Can't proceed with Tika for PDFs.")
            # You can decide if you want to fail or skip PDF.

        try:
            bearer_token = google_auth.impersonated_id_token(
                serverurl=SERVER_DOMAIN
            ).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token:")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Failed to obtain impersonated ID token.", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        # We'll store tasks for XLSX in a list => so we can run them in parallel
        xlsx_tasks = []

        for blob, blob_name, last_modified in to_process:
            file_key = generate_md5_hash(
                sub_for_hash, project_id, data_source_id, blob_name
            )
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(blob_name)

            # If file not in DB, create record
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
                    source_type="azureBlob",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(
                    file_key, status="processing", updatedAt=now_dt.isoformat()
                )

            extension = "pdf"
            if "." in blob_name:
                extension = blob_name.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")
            try:
                with open(local_tmp_path, "wb") as f:
                    download_stream = container_client.download_blob(blob.name)
                    f.write(download_stream.readall())
            except Exception as e:
                log.exception(f"Failed to download azure blob: {blob_name}")
                psql.update_file_by_id(
                    file_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

            # If it's XLSX, queue up a parallel task
            if extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                # We'll *not* process immediately. We'll add an async task for later concurrency
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
                            sourceType="azureBlob",
                        )
                    )
                except Exception as e:
                    log.exception(f"Failed Tika/embedding for {base_name}:")
                    psql.update_file_by_id(
                        file_key, status="failed", updatedAt=now_dt.isoformat()
                    )
                    continue

                # Mark processed
                psql.update_file_by_id(
                    file_key,
                    status="processed",
                    pageCount=len(results),
                    updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
                )

                # usage
                used_credits = len(results) * 1.5
                if sub_id:
                    message = json.dumps(
                        {
                            "subscription_id": sub_id,
                            "data_source_id": data_source_id,
                            "project_id": project_id,
                            "creditsUsed": used_credits,
                        }
                    )
                    try:
                        google_pub_sub.publish_messages_with_retry_settings(
                            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                        )
                    except Exception as e:
                        log.warning(
                            f"Publish usage failed for Azure file {base_name}: {str(e)}"
                        )

            else:
                # If it's some unknown extension => mark failed or skip
                log.warning(
                    f"Skipping unsupported extension {extension} for {base_name}"
                )
                psql.update_file_by_id(
                    file_key, status="failed", updatedAt=now_dt.isoformat()
                )
                if os.path.exists(local_tmp_path):
                    os.remove(local_tmp_path)

        # ----------------------------
        # PARALLEL XLSX PROCESSING
        # ----------------------------
        # We'll run them all in an async gather
        # Let's define an async function that handles them
        async def process_all_xlsx_tasks(xlsx_files):
            tasks = []
            for f_key, b_name, tmp_path, last_modified in xlsx_files:
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
            # gather them all
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        if xlsx_tasks:
            log.info(f"Processing {len(xlsx_tasks)} XLSX files in parallel.")
            xlsx_results = loop.run_until_complete(process_all_xlsx_tasks(xlsx_tasks))

            # Now xlsx_results is a list of tuples => (final_status, usage_credits, error_msg)
            # in the same order as xlsx_tasks
            for (f_key, b_name, _, last_modified), (
                final_status,
                usage_credits,
                error_msg,
            ) in zip(xlsx_tasks, xlsx_results):
                now_dt = datetime.now(CENTRAL_TZ)
                if final_status == "processed":
                    psql.update_file_by_id(
                        f_key, status="processed", updatedAt=now_dt.isoformat()
                    )
                    # publish usage if usage_credits>0
                    if usage_credits > 0 and sub_id:
                        message = json.dumps(
                            {
                                "subscription_id": sub_id,
                                "data_source_id": data_source_id,
                                "project_id": project_id,
                                "creditsUsed": usage_credits,
                            }
                        )
                        try:
                            google_pub_sub.publish_messages_with_retry_settings(
                                GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                            )
                        except Exception as e:
                            log.warning(
                                f"Publish usage failed for XLSX file {b_name}: {str(e)}"
                            )
                else:
                    # Mark failed
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
        log.exception("Error in Azure ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


def shutdown_handler(sig, frame):
    log.info(f"Caught signal {signal.strsignal(sig)}. Exiting.")
    sys.exit(0)


if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            log.info(f"Created UPLOADS_FOLDER at {UPLOADS_FOLDER}")
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)
    log.debug(f"UPLOADS_FOLDER is set to: {UPLOADS_FOLDER}")

    if not SERVER_URL:
        log.error("SERVER_URL environment variable is not set.")
        # If you rely on Tika for docx/pptx, this is a problem. For XLSX we call a different endpoint.
        # sys.exit(1)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
