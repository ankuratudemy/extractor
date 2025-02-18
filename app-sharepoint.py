#!/usr/bin/env python3
# sharepoint_ingest.py

import os
import io
import sys
import json
import requests
import asyncio
import aiohttp
import subprocess
from datetime import datetime, timedelta
from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, file_processor, google_auth

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

SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET")


def always_refresh_sharepoint_token(ds):
    refresh_token = ds.get("sharePointRefreshToken")
    if not refresh_token:
        log.warning(f"No sharePointRefreshToken for DS {ds['id']}")
        return ds

    token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    body_params = {
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": "https://graph.microsoft.com/.default offline_access",
    }
    try:
        resp = requests.post(token_url, data=body_params)
        if not resp.ok:
            log.error(
                f"[always_refresh_sharepoint_token] Refresh request failed: {resp.text}"
            )
            return ds
        td = resp.json()
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"No access_token in response: {td}")
            return ds

        expires_in = td.get("expires_in", 3920)
        now_dt = datetime.now(CENTRAL_TZ)
        new_expires_dt = now_dt + timedelta(seconds=expires_in)
        new_expires_str = new_expires_dt.isoformat()

        psql.update_data_source_by_id(
            ds["id"],
            sharePointAccessToken=new_access_token,
            sharePointExpiresAt=new_expires_str,
        )
        ds["sharePointAccessToken"] = new_access_token
        ds["sharePointExpiresAt"] = new_expires_str
        return ds
    except Exception as e:
        log.exception("[always_refresh_sharepoint_token] Error refreshing token:")
        return ds


def run_job():
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"SharePoint ingest => event_data: {event_data}")
    except Exception as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if not ds:
            log.error(f"No DS found for id={data_source_id}")
            psql.update_data_source_by_id(data_source_id, status="failed")
            sys.exit("DS not found")

        if ds.get("sourceType") != "sharepoint":
            log.error(f"DS {data_source_id} is {ds.get('sourceType')}, not sharepoint.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            sys.exit("Invalid data source type")

        ds = always_refresh_sharepoint_token(ds)
        access_token = ds.get("sharePointAccessToken")
        if not access_token:
            psql.update_data_source_by_id(data_source_id, status="failed")
            sys.exit("No valid sharePointAccessToken after refresh")

        site_id = ds.get("sharePointSiteId")
        list_id = ds.get("sharePointListId")
        folder_id = ds.get("sharePointFolderId")
        if not site_id or not (list_id or folder_id):
            psql.update_data_source_by_id(data_source_id, status="failed")
            sys.exit("Missing site_id or list/folder ID")

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
        from shared.sharepoint_helpers import list_all_sharepoint_files_fallback

        is_list = bool(list_id)
        list_or_folder_id = list_id if is_list else folder_id
        try:
            sharepoint_files = list_all_sharepoint_files_fallback(
                access_token, site_id, list_or_folder_id, is_list
            )
        except Exception as e:
            log.exception("Failed listing files from SharePoint:")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return (str(e), 500)

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
        sharepoint_file_keys = set()
        spFileId_to_key = {}

        for item in sharepoint_files:
            item_id = item["id"]
            item_name = item.get("name", "untitled")
            item_folder = item.get("folder")
            item_modified_time = item.get("lastModifiedDateTime")

            if item_folder:
                continue

            file_key = generate_md5_hash(
                sub_for_hash, project_id, data_source_id, item_id
            )
            sharepoint_file_keys.add(file_key)
            spFileId_to_key[item_id] = file_key

            if file_key not in db_file_keys:
                new_files.append(item)
            else:
                if last_sync_dt and item_modified_time:
                    try:
                        item_modified_dt = ensure_timezone_aware(
                            parser.isoparse(item_modified_time)
                        )
                    except:
                        item_modified_dt = None
                    if item_modified_dt and item_modified_dt > last_sync_dt:
                        updated_files.append(item)
                else:
                    updated_files.append(item)

        removed_keys = db_file_keys - sharepoint_file_keys
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
            return ("No new/updated files", 200)

        if not SERVER_URL:
            log.error("SERVER_URL not set => can't proceed with Tika.")
            psql.update_data_source_by_id(data_source_id, status="failed")
            return ("No SERVER_URL", 500)

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

        for sp_item in to_process:
            item_id = sp_item["id"]
            item_name = sp_item.get("name", "untitled")
            file_key = spFileId_to_key[item_id]
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
                    source_type="sharepoint",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(
                    file_key, status="processing", updatedAt=now_dt.isoformat()
                )

            # Download
            from shared.sharepoint_helpers import download_file_with_fallback

            try:
                content_bytes = download_file_with_fallback(
                    access_token, site_id, item_id, list_or_folder_id
                )
            except Exception as e:
                log.exception(f"Failed to download {item_name} from SharePoint:")
                psql.update_file_by_id(
                    file_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

            if not content_bytes:
                log.error(f"Item {item_name} => empty or invalid content.")
                psql.update_file_by_id(
                    file_key, status="not supported", updatedAt=now_dt.isoformat()
                )
                continue

            # Guess extension
            extension = "pdf"
            if "." in item_name:
                extension = item_name.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")
            try:
                with open(local_tmp_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing {item_name} to disk:")
                psql.update_file_by_id(
                    file_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

            if extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                xlsx_tasks.append((file_key, item_name, local_tmp_path, now_dt))
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
                    log.error(f"Failed processing {item_name} as {extension}: {str(e)}")
                    psql.update_file_by_id(
                        file_key, status="failed", updatedAt=now_dt.isoformat()
                    )
                    if os.path.exists(local_tmp_path):
                        os.remove(local_tmp_path)
                    continue

                if os.path.exists(local_tmp_path):
                    os.remove(local_tmp_path)

            # Tika
            try:
                results = loop.run_until_complete(
                    process_pages_async(
                        final_pages,
                        headers,
                        item_name,
                        project_id,
                        file_key,
                        data_source_id,
                        last_modified=now_dt,
                        sourceType="sharepoint",
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding for {item_name}:")
                psql.update_file_by_id(
                    file_key, status="failed", updatedAt=now_dt.isoformat()
                )
                continue

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
                    log.warning(
                        f"Publish usage failed for SharePoint file {item_name}: {e}"
                    )

        # XLSX parallel
        if xlsx_tasks:
            log.info(
                f"Processing {len(xlsx_tasks)} XLSX files in parallel for SharePoint."
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
        log.exception("Error in SharePoint ingestion flow:")
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
