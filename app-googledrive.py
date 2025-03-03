#!/usr/bin/env python3
# google_drive_ingest.py

import os
import io
import sys
import json
import asyncio
import requests
import signal
from datetime import datetime, timedelta
from dateutil import parser

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
    UPLOADS_FOLDER,
)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

def always_refresh_drive_token(ds):
    refresh_token = ds.get("googleRefreshToken")
    if not refresh_token:
        log.warning(f"No refresh token for DS {ds['id']}; skipping.")
        return ds

    token_url = "https://oauth2.googleapis.com/token"
    body_params = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        resp = requests.post(token_url, data=body_params)
        if not resp.ok:
            log.error(f"[always_refresh_drive_token] Refresh request failed: {resp.text}")
            return ds
        td = resp.json()
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"[always_refresh_drive_token] No access_token in response: {td}")
            return ds

        expires_in = td.get("expires_in", 3920)
        now_dt = datetime.now(CENTRAL_TZ)
        new_expires_dt = now_dt + timedelta(seconds=expires_in)
        psql.update_data_source_by_id(
            ds["id"],
            googleAccessToken=new_access_token,
            googleExpiresAt=new_expires_dt.isoformat(),
        )
        ds["googleAccessToken"] = new_access_token
        ds["googleExpiresAt"] = new_expires_dt.isoformat()
        return ds
    except Exception as e:
        log.exception("[always_refresh_drive_token] Error refreshing token:")
        return ds

def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"Google Drive ingest => event_data: {event_data}")
    except Exception:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            return (f"No DataSource for id={data_source_id}", 404)
        if ds["sourceType"] != "googleDrive":
            return (f"DataSource {data_source_id} is not googleDrive.", 400)

        ds = always_refresh_drive_token(ds)
        access_token = ds.get("googleAccessToken")
        if not access_token:
            return ("No valid googleAccessToken after refresh", 400)

        folder_id = ds.get("googleDriveFolderId")
        if not folder_id:
            return ("No googleDriveFolderId in DS record", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = parse_last_sync_time(last_sync_time)

        from shared.drive_helpers import list_all_files_recursively_with_retry, download_drive_file_content

        drive_files = list_all_files_recursively_with_retry(access_token, folder_id, ds)

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
        drive_file_keys = set()

        for gf in drive_files:
            gf_id = gf["id"]
            gf_name = gf["name"]
            gf_mime = gf.get("mimeType")
            gf_modified_time = gf.get("modifiedTime")
            if "folder" in gf_mime:
                continue

            gf_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gf_id)
            drive_file_keys.add(gf_key)

            if gf_key not in db_file_keys:
                new_files.append(gf)
            else:
                if last_sync_dt and gf_modified_time:
                    gf_mod_dt = parse_last_sync_time(gf_modified_time)
                    if gf_mod_dt and gf_mod_dt > last_sync_dt:
                        updated_files.append(gf)
                else:
                    updated_files.append(gf)

        remove_missing_files(db_file_keys, drive_file_keys, data_source_id, project_id, namespace=project_id)

        to_process = new_files + updated_files
        if not to_process:
            finalize_data_source(data_source_id, ds, new_status="processed")
            return ("No new or updated files", 200)

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        xlsx_tasks = []

        for gf in to_process:
            gf_id = gf["id"]
            gf_name = gf["name"]
            gf_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gf_id)
            now_dt = datetime.now(CENTRAL_TZ)

            if gf_key not in db_file_keys_list:
                psql.add_new_file(
                    file_id=gf_key,
                    status="processing",
                    created_at=now_dt,
                    updated_at=now_dt,
                    name=gf_name,
                    file_type="unknown",
                    file_key=f"{sub_for_hash}-{project_id}-{gf_key}",
                    page_count=0,
                    upload_url="None",
                    source_type="googleDrive",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(gf_key)
            else:
                psql.update_file_by_id(gf_key, status="processing", updatedAt=now_dt.isoformat())

            try:
                content_bytes, extension = download_drive_file_content(
                    access_token,
                    gf_id,
                    mime_type=gf.get("mimeType")
                )
            except Exception as e:
                log.exception(f"Failed to download {gf_name}:")
                psql.update_file_by_id(gf_key, status="failed")
                continue

            if not content_bytes or not extension:
                psql.update_file_by_id(gf_key, status="not supported")
                continue

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{gf_key}.{extension}")
            try:
                with open(local_tmp_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Failed writing file {gf_name} to disk:")
                psql.update_file_by_id(gf_key, status="failed")
                continue

            if extension.lower() in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((gf_key, gf_name, local_tmp_path, now_dt))
                continue

            final_status, page_count, error_msg = process_file_with_tika(
                file_key=gf_key,
                base_name=gf_name,
                local_tmp_path=local_tmp_path,
                extension=extension.lower(),
                project_id=project_id,
                data_source_id=data_source_id,
                last_modified=now_dt,
                source_type="googledrive",
                sub_id=sub_id
            )

            if final_status == "failed":
                log.error(f"File {gf_name} failed: {error_msg}")

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
                    log.error(f"XLSX processing failed for {b_name}. {error_msg}")
                    psql.update_file_by_id(f_key, status="failed")
                else:
                    log.info(f"XLSX file {b_name} processed successfully.")
                    psql.update_file_by_id(f_key, status="processed")

        finalize_data_source(data_source_id, ds, new_status="processed")
        return ("OK", 200)

    except Exception as e:
        log.exception("Error in googleDrive ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER: {e}")
            sys.exit(1)

    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
