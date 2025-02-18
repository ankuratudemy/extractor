#!/usr/bin/env python3
# google_drive_ingest.py

import os
import io
import sys
import json
import asyncio
import aiohttp
import requests
import subprocess
from datetime import datetime, timedelta
from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor

# Import from `common_code.py`
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
            log.error(
                f"[always_refresh_drive_token] Refresh request failed: {resp.text}"
            )
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
    except Exception as e:
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
            return (
                f"DataSource {data_source_id} is {ds['sourceType']}, not googleDrive.",
                400,
            )

        # Always refresh
        ds = always_refresh_drive_token(ds)
        access_token = ds.get("googleAccessToken")
        if not access_token:
            return ("No valid googleAccessToken after refresh", 400)

        folder_id = ds.get("googleDriveFolderId")
        if not folder_id:
            return ("No googleDriveFolderId in DS record", 400)

        # last sync
        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None

        # List files
        from shared.drive_helpers import (
            list_all_files_recursively_with_retry,
        )  # or inline

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
        driveFileId_to_key = set()
        drive_file_keys = set()

        # Distinguish new vs updated
        for gf in drive_files:
            gf_id = gf["id"]
            gf_name = gf["name"]
            gf_mime = gf.get("mimeType")
            gf_modified_time = gf.get("modifiedTime")

            if "folder" in gf_mime:
                # skip folders
                continue

            gf_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gf_id)
            drive_file_keys.add(gf_key)

            if gf_key not in db_file_keys:
                new_files.append(gf)
            else:
                if last_sync_dt:
                    try:
                        gf_mod_dt = ensure_timezone_aware(
                            parser.isoparse(gf_modified_time)
                        )
                    except:
                        gf_mod_dt = None
                    if gf_mod_dt and gf_mod_dt > last_sync_dt:
                        updated_files.append(gf)
                else:
                    updated_files.append(gf)

        removed_keys = db_file_keys - drive_file_keys
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(
                r_key, data_source_id, project_id, project_id
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
            log.error("SERVER_URL is not set for Tika.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("SERVER_URL is not set", 500)

        # get bearer token for Tika
        try:
            bearer_token = google_auth.impersonated_id_token(
                serverurl=SERVER_DOMAIN
            ).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token for Tika:")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("Failed to get Tika token", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()

        db_file_keys_list = set(db_file_keys)

        # XLSX tasks
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
                psql.update_file_by_id(
                    gf_key, status="processing", updatedAt=now_dt.isoformat()
                )

            # Download logic (export, alt=media, etc.)
            from shared.drive_helpers import download_drive_file_content

            try:
                content_bytes, extension = download_drive_file_content(
                    access_token, gf_id, mime_type=gf.get("mimeType")
                )
            except Exception as e:
                log.exception(f"Failed to download {gf_name}:")
                psql.update_file_by_id(
                    gf_key, status="failed", updatedAt=now_dt.isoformat()
                )
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
                psql.update_file_by_id(
                    gf_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

            if extension.lower() in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((gf_key, gf_name, local_tmp_path, now_dt))
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
                    log.error(f"Failed processing {gf_name} => {e}: {str(e)}")
                    psql.update_file_by_id(
                        gf_key, status="failed", updatedAt=now_dt.isoformat()
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
                        gf_name,
                        project_id,
                        gf_key,
                        data_source_id,
                        last_modified=now_dt,
                        sourceType="googledrive"
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding for {gf_name}:")
                psql.update_file_by_id(
                    gf_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

            psql.update_file_by_id(gf_key, status="processed", pageCount=len(results))

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
                    log.warning(
                        f"Publish usage failed for Google Drive file {gf_name}: {e}"
                    )

        # XLSX parallel
        if xlsx_tasks:
            log.info(
                f"Processing {len(xlsx_tasks)} XLSX files in parallel for Google Drive."
            )

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
        log.exception("Error in googleDrive ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (str(e), 500)


if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER: {e}")
            sys.exit(1)

    if not SERVER_URL:
        log.error("SERVER_URL environment variable is not set.")
        sys.exit(1)

    run_job()
else:
    import signal

    signal.signal(signal.SIGTERM, shutdown_handler)
