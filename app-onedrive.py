#!/usr/bin/env python3
# onedrive_ingest.py

import os
import sys
import json
import requests
import asyncio
import aiohttp
import subprocess
from datetime import datetime, timedelta
from typing import List, Tuple
from zoneinfo import ZoneInfo
from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, file_processor, google_auth

# Import the shared code from common_code.py
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
    process_xlsx_blob,   # <-- Import our XLSX helper
    shutdown_handler,    # If you want to reuse the same signal handler
)

ONEDRIVE_CLIENT_ID = os.environ.get("ONEDRIVE_CLIENT_ID")
ONEDRIVE_CLIENT_SECRET = os.environ.get("ONEDRIVE_CLIENT_SECRET")


def always_refresh_onedrive_token(ds):
    """
    Refresh the OneDrive token using the refresh token, ignoring expiresAt logic.
    """
    refresh_token = ds.get("oneDriveRefreshToken")
    if not refresh_token:
        log.warning(f"No oneDriveRefreshToken for DS {ds['id']}; skipping refresh.")
        return ds

    token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    body_params = {
        "client_id": ONEDRIVE_CLIENT_ID,
        "client_secret": ONEDRIVE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post(token_url, data=body_params)
        if not resp.ok:
            log.error(f"[always_refresh_onedrive_token] Refresh request failed: {resp.text}")
            return ds
        td = resp.json()
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"[always_refresh_onedrive_token] No access_token in response: {td}")
            return ds

        expires_in = td.get("expires_in", 3920)
        now_dt = datetime.now(CENTRAL_TZ)
        new_expires_dt = now_dt + timedelta(seconds=expires_in)
        new_expires_str = new_expires_dt.isoformat()
        psql.update_data_source_by_id(
            ds["id"],
            oneDriveAccessToken=new_access_token,
            oneDriveExpiresAt=new_expires_str,
        )
        ds["oneDriveAccessToken"] = new_access_token
        ds["oneDriveExpiresAt"] = new_expires_str
        log.info(f"[always_refresh_onedrive_token] Refreshed token for DS {ds['id']}, expiresAt={new_expires_dt}")
        return ds
    except Exception as e:
        log.exception("[always_refresh_onedrive_token] Error refreshing token for DS:")
        return ds


def _list_all_onedrive_files_recursive(token: str, folder_id: str, ds) -> List[dict]:
    """
    Recursively list all files in OneDrive using MS Graph.
    """
    results = []
    stack = [folder_id]
    base_url = "https://graph.microsoft.com/v1.0/me/drive/items"
    headers = {"Authorization": f"Bearer {token}"}

    while stack:
        current_folder = stack.pop()
        page_url = f"{base_url}/{current_folder}/children?$top=200"
        while page_url:
            log.debug(f"[_list_all_onedrive_files_recursive] GET => {page_url}")
            resp = requests.get(page_url, headers=headers)
            if not resp.ok:
                log.error(
                    f"[_list_all_onedrive_files_recursive] List failed => "
                    f"{resp.status_code} - {resp.text[:500]}"
                )
                if resp.status_code == 401:
                    raise ValueError("401 Unauthorized - token refresh needed.")
                resp.raise_for_status()

            data = resp.json()
            items = data.get("value", [])
            for item in items:
                # If item is a folder, push onto stack to go deeper
                if item.get("folder"):
                    stack.append(item["id"])
                results.append(item)

            page_url = data.get("@odata.nextLink")

    return results


def list_all_onedrive_files_recursively_with_retry(access_token: str, folder_id: str, ds) -> List[dict]:
    """
    Wraps the recursive listing and handles exceptions.
    """
    try:
        od_files = _list_all_onedrive_files_recursive(access_token, folder_id, ds)
        return od_files
    except Exception as e:
        log.error(f"Failed to list OneDrive files => {e}")
        raise


def download_onedrive_file_content(access_token, item_id):
    """
    Downloads the OneDrive file contents via the Graph endpoint.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content"

    resp = requests.get(url, headers=headers, stream=True)
    if not resp.ok:
        log.error(
            f"[download_onedrive_file_content] Could not download => "
            f"{resp.status_code} - {resp.text[:300]}"
        )
        resp.raise_for_status()

    return resp.content


def run_job():
    """
    Main OneDrive ingestion flow, including XLSX handling in parallel.
    """
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"OneDrive ingest: parsed event_data => {event_data}")
    except Exception as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        log.info(f"data source from db => {ds}")
        if not ds:
            log.info(f"No DS found for id={data_source_id}")
            return (f"No DataSource found for id={data_source_id}", 404)
        if ds["sourceType"] != "oneDrive":
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not oneDrive.", 400)

        # 1) Refresh the OneDrive token
        ds = always_refresh_onedrive_token(ds)
        access_token = ds.get("oneDriveAccessToken")
        if not access_token:
            return ("No valid oneDriveAccessToken after refresh", 400)

        folder_id = ds.get("oneDriveFolderId", "root")
        log.info(f"OneDrive folder_id => {folder_id}")
        if not folder_id:
            return ("No oneDriveFolderId in DS record", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None
        log.info(f"last_sync_dt => {last_sync_dt}")

        # 2) List files in OneDrive
        try:
            od_files = list_all_onedrive_files_recursively_with_retry(access_token, folder_id, ds)
        except Exception as e:
            log.exception("Failed listing files from OneDrive after token refresh:")
            psql.update_data_source_by_id(data_source_id, status="error")
            return (str(e), 500)

        log.info(f"Found {len(od_files)} total items in folder {folder_id}")

        existing_files = psql.fetch_files_by_data_source_id(data_source_id)
        db_file_keys = set(ef["id"] for ef in existing_files)

        new_files = []
        updated_files = []

        sub_id = None
        if project_id:
            proj_details = psql.get_project_details(project_id)
            if proj_details:
                sub_id = proj_details.get("subscriptionId")
        sub_for_hash = sub_id if sub_id else "no_subscription"

        odFileId_to_key = {}
        od_file_keys = set()

        # Classify new vs. updated
        for item in od_files:
            item_id = item["id"]
            item_name = item.get("name", "untitled")
            item_file = item.get("file")
            item_folder = item.get("folder")
            item_modified_time = item.get("lastModifiedDateTime")

            # skip folders (we recursively processed them anyway)
            if item_folder:
                continue

            # derive unique file key
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, item_id)
            odFileId_to_key[item_id] = file_key
            od_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append(item)
            else:
                if last_sync_dt:
                    try:
                        item_modified_dt = ensure_timezone_aware(parser.isoparse(item_modified_time))
                    except:
                        item_modified_dt = None
                    if item_modified_dt and item_modified_dt > last_sync_dt:
                        updated_files.append(item)
                else:
                    updated_files.append(item)

        removed_keys = db_file_keys - od_file_keys
        log.info(f"Removed keys => {removed_keys}")
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(r_key, data_source_id, project_id, namespace=project_id)

        log.info(f"New files => {len(new_files)}, Updated => {len(updated_files)}")
        to_process = new_files + updated_files
        if not to_process:
            log.info("No new/updated files => done.")
            now_dt = datetime.now(CENTRAL_TZ)
            psql.update_data_source_by_id(
                data_source_id,
                status="processed",
                lastSyncTime=now_dt.isoformat(),
                nextSyncTime=compute_next_sync_time(now_dt, ds.get("syncOption")).isoformat(),
            )
            return ("No new/updated files", 200)

        if not SERVER_URL:
            log.error("SERVER_URL not set => can't proceed with Tika.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("No SERVER_URL", 500)

        # Get Bearer token for Tika server
        try:
            bearer_token = google_auth.impersonated_id_token(serverurl=SERVER_DOMAIN).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token for Tika:")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Failed to get Tika token", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        # We'll store XLSX tasks for parallel processing
        xlsx_tasks = []

        # 3) Download and process each file
        for od_item in to_process:
            item_id = od_item["id"]
            item_name = od_item.get("name", "untitled")
            item_modified_time = od_item.get("lastModifiedDateTime", "")
            file_key = odFileId_to_key[item_id]

            now_dt = datetime.now(CENTRAL_TZ)
            if file_key not in db_file_keys_list:
                psql.add_new_file(
                    file_id=file_key,
                    status="processing",
                    created_at=now_dt,
                    updated_at=now_dt,
                    name=item_name,
                    file_type="unknown",
                    file_key=f"{sub_for_hash}-{project_id}-{file_key}",
                    page_count=0,
                    upload_url="None",
                    source_type="oneDrive",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            # Download from OneDrive
            try:
                content_bytes = download_onedrive_file_content(access_token, item_id)
            except Exception as e:
                log.exception(f"Failed to download {item_name} from OneDrive:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            if not content_bytes:
                log.error(f"Item {item_name} => empty or invalid content.")
                psql.update_file_by_id(file_key, status="not supported", updatedAt=now_dt.isoformat())
                continue

            # Guess extension from item_name (or item['file']['mimeType'])
            extension = "pdf"
            if "." in item_name:
                extension = item_name.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")
            try:
                with open(local_tmp_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing OneDrive file {item_name} to disk:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            # If it's XLSX, let's queue the parallel processing
            if extension == "xlsx":
                # We'll skip immediate Tika flow; queue an async XLSX ingestion
                xlsx_tasks.append((file_key, item_name, local_tmp_path, now_dt))
                continue

            # Otherwise proceed w/ PDF / docx / pptx flow:
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
                        raise ValueError("Conversion to PDF failed")
                raise ValueError(f"Unsupported file format {file_ext} for OneDrive logic")

            try:
                final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
            except Exception as e:
                log.error(f"Failed processing {item_name} => {extension}: {str(e)}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                if os.path.exists(local_tmp_path):
                    os.remove(local_tmp_path)
                continue

            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

            # Tika / embedding
            try:
                results = loop.run_until_complete(
                    process_pages_async(
                        final_pages,
                        headers,
                        item_name,
                        project_id,
                        file_key,
                        data_source_id,
                        last_modified=now_dt,  # or parse item_modified_time
                        sourceType="oneDrive"
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding step for {item_name}:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            psql.update_file_by_id(
                file_key,
                status="processed",
                pageCount=len(results),
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )

            used_credits = len(results) * 1.5
            if sub_id:
                msg = json.dumps({
                    "subscription_id": sub_id,
                    "data_source_id": data_source_id,
                    "project_id": project_id,
                    "creditsUsed": used_credits
                })
                try:
                    google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=msg)
                except Exception as e:
                    log.warning(f"Publish usage failed for OneDrive file {item_name}: {e}")

        # Now handle XLSX tasks in parallel
        if xlsx_tasks:
            log.info(f"Processing {len(xlsx_tasks)} XLSX files in parallel for OneDrive.")

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

            # xlsx_results => list of (final_status, usage_credits, error_msg)
            for ((f_key, b_name, _, lm), (final_status, usage_credits, error_msg)) in zip(xlsx_tasks, xlsx_results):
                now_dt = datetime.now(CENTRAL_TZ)
                if final_status == "processed":
                    psql.update_file_by_id(f_key, status="processed", updatedAt=now_dt.isoformat())
                    if usage_credits > 0 and sub_id:
                        msg = json.dumps({
                            "subscription_id": sub_id,
                            "data_source_id": data_source_id,
                            "project_id": project_id,
                            "creditsUsed": usage_credits
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
        log.exception("Error in OneDrive ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="error")
        return (str(e), 500)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            log.info(f"Created UPLOADS_FOLDER at {UPLOADS_FOLDER}")
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)

    if not SERVER_URL:
        log.error("SERVER_URL environment variable is not set.")
        sys.exit(1)

    run_job()
else:
    import signal
    signal.signal(signal.SIGTERM, shutdown_handler)
