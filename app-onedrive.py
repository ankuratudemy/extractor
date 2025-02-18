#!/usr/bin/env python3
# onedrive_ingest.py

import os
import sys
import json
import requests
import asyncio
import signal
from datetime import datetime, timedelta
from dateutil import parser

from shared.logging_config import log
from shared import psql, google_pub_sub, file_processor, google_auth
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
    UPLOADS_FOLDER,
)

ONEDRIVE_CLIENT_ID = os.environ.get("ONEDRIVE_CLIENT_ID")
ONEDRIVE_CLIENT_SECRET = os.environ.get("ONEDRIVE_CLIENT_SECRET")


def always_refresh_onedrive_token(ds):
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
            log.error(f"No access_token in response: {td}")
            return ds

        expires_in = td.get("expires_in", 3920)
        now_dt = datetime.now(CENTRAL_TZ)
        new_expires_dt = now_dt + timedelta(seconds=expires_in)
        psql.update_data_source_by_id(
            ds["id"],
            oneDriveAccessToken=new_access_token,
            oneDriveExpiresAt=new_expires_dt.isoformat(),
        )
        ds["oneDriveAccessToken"] = new_access_token
        ds["oneDriveExpiresAt"] = new_expires_dt.isoformat()
        log.info(f"Refreshed OneDrive token for DS {ds['id']}.")
        return ds
    except Exception as e:
        log.exception("[always_refresh_onedrive_token] Error refreshing token for DS:")
        return ds


def _list_all_onedrive_files_recursive(token, folder_id, ds):
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
            resp = requests.get(page_url, headers=headers)
            if not resp.ok:
                log.error(f"[_list_all_onedrive_files_recursive] {resp.status_code} {resp.text}")
                if resp.status_code == 401:
                    raise ValueError("401 Unauthorized - token refresh needed.")
                resp.raise_for_status()

            data = resp.json()
            items = data.get("value", [])
            for item in items:
                if item.get("folder"):
                    stack.append(item["id"])
                results.append(item)

            page_url = data.get("@odata.nextLink")

    return results


def list_all_onedrive_files_recursively_with_retry(access_token, folder_id, ds):
    try:
        return _list_all_onedrive_files_recursive(access_token, folder_id, ds)
    except Exception as e:
        log.error(f"Failed to list OneDrive files => {e}")
        raise


def download_onedrive_file_content(access_token, item_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content"
    resp = requests.get(url, headers=headers, stream=True)
    if not resp.ok:
        log.error(f"[download_onedrive_file_content] {resp.status_code} - {resp.text}")
        resp.raise_for_status()
    return resp.content


def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"OneDrive ingest: parsed event_data => {event_data}")
    except Exception:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            return (f"No DS found for id={data_source_id}", 404)
        if ds["sourceType"] != "oneDrive":
            return (f"DS {data_source_id} is not oneDrive.", 400)

        ds = always_refresh_onedrive_token(ds)
        access_token = ds.get("oneDriveAccessToken")
        if not access_token:
            return ("No valid oneDriveAccessToken after refresh", 400)

        folder_id = ds.get("oneDriveFolderId", "root")
        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = parse_last_sync_time(last_sync_time)

        od_files = list_all_onedrive_files_recursively_with_retry(access_token, folder_id, ds)
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

        od_file_keys = set()
        for item in od_files:
            item_id = item["id"]
            item_name = item.get("name", "untitled")
            item_folder = item.get("folder")
            item_modified_time = item.get("lastModifiedDateTime")

            if item_folder:
                continue

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, item_id)
            od_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append(item)
            else:
                if last_sync_dt and item_modified_time:
                    try:
                        item_modified_dt = parse_last_sync_time(item_modified_time)
                    except:
                        item_modified_dt = None
                    if item_modified_dt and item_modified_dt > last_sync_dt:
                        updated_files.append(item)
                else:
                    updated_files.append(item)

        remove_missing_files(db_file_keys, od_file_keys, data_source_id, project_id, namespace=project_id)

        to_process = new_files + updated_files
        if not to_process:
            finalize_data_source(data_source_id, ds, new_status="processed")
            return ("No new/updated files", 200)

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        xlsx_tasks = []

        for od_item in to_process:
            item_id = od_item["id"]
            item_name = od_item.get("name", "untitled")
            item_modified_time = od_item.get("lastModifiedDateTime", "")
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, item_id)

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

            # Download
            try:
                content_bytes = download_onedrive_file_content(access_token, item_id)
            except Exception as e:
                log.exception(f"Failed to download {item_name} from OneDrive:")
                psql.update_file_by_id(file_key, status="failed")
                continue

            if not content_bytes:
                log.error(f"Item {item_name} => empty/invalid.")
                psql.update_file_by_id(file_key, status="not supported")
                continue

            extension = "pdf"
            if "." in item_name:
                extension = item_name.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")
            try:
                with open(local_tmp_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing OneDrive file {item_name} to disk:")
                psql.update_file_by_id(file_key, status="failed")
                continue

            if extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((file_key, item_name, local_tmp_path, now_dt))
                continue

            final_status, page_count, error_msg = process_file_with_tika(
                file_key=file_key,
                base_name=item_name,
                local_tmp_path=local_tmp_path,
                extension=extension,
                project_id=project_id,
                data_source_id=data_source_id,
                last_modified=now_dt,
                source_type="oneDrive",
                sub_id=sub_id
            )

            if final_status == "failed":
                log.error(f"OneDrive file {item_name} failed: {error_msg}")

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
        log.exception("Error in OneDrive ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            log.info(f"Created UPLOADS_FOLDER at {UPLOADS_FOLDER}")
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER: {e}")
            sys.exit(1)

    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
