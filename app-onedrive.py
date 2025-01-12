#!/usr/bin/env python3
# onedrive_ingest.py

import os
import sys
import json
import io
import time
import signal
import ssl
import hashlib
import requests
import asyncio
import aiohttp
import tiktoken
import subprocess
from datetime import datetime, timedelta
from typing import List, Tuple
from zoneinfo import ZoneInfo

from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, file_processor  # or rename as needed
from pinecone.grpc import PineconeGRPC as Pinecone

# If using Vertex AI for embeddings
from vertexai.preview.language_models import TextEmbeddingModel

# ----------------------------------------------------------------------------
# ENV VARIABLES
# ----------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER", "/tmp/uploads")  # Default to /tmp/uploads if not set

SERVER_DOMAIN = os.environ.get("SERVER_URL")  # e.g., mydomain.com
SERVER_URL = f"https://{SERVER_DOMAIN}/tika" if SERVER_DOMAIN else None

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

ONEDRIVE_CLIENT_ID = os.environ.get("ONEDRIVE_CLIENT_ID")
ONEDRIVE_CLIENT_SECRET = os.environ.get("ONEDRIVE_CLIENT_SECRET")

# Example: For Microsoft Graph, store tokens:
# ds["oneDriveAccessToken"] / ds["oneDriveRefreshToken"]

# Define Central US Timezone
CENTRAL_TZ = ZoneInfo("America/Chicago")

# ----------------------------------------------------------------------------
# INITIALIZE PINECONE
# ----------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------


def ensure_timezone_aware(dt):
    """Ensures that the datetime object is timezone-aware (Central US Time by default)."""
    if dt and dt.tzinfo is None:
        return dt.replace(tzinfo=CENTRAL_TZ)
    return dt


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def generate_md5_hash(*args):
    """
    Generates an MD5 hash from the JSON-dumped data
    ensuring uniqueness for user-specific combos.
    """
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    md5_hash = hashlib.md5(combined_string.encode("utf-8")).hexdigest()
    return md5_hash


def computeNextSyncTime(from_date: datetime, sync_option: str = None) -> datetime:
    """
    Computes the next sync time based on sync_option: 'hourly'|'daily'|'weekly'|'monthly'.
    Fallback to 1 hour if no valid syncOption is provided.
    """
    if not isinstance(from_date, datetime):
        from_date = datetime.now(CENTRAL_TZ)

    if sync_option is None:
        sync_option = "hourly"

    if sync_option == "hourly":
        return from_date + timedelta(hours=1)
    elif sync_option == "daily":
        return from_date + timedelta(days=1)
    elif sync_option == "weekly":
        return from_date + timedelta(weeks=1)
    elif sync_option == "monthly":
        return from_date + timedelta(days=30)
    else:
        return from_date + timedelta(hours=1)


def remove_file_from_db_and_pinecone(file_id, ds_id, project_id, namespace):
    """
    Removes references in DB + Pinecone for the given file_id => consistent with GoogleDrive logic.
    """
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        # Example: We'll assume a prefix-based list for Pinecone
        for ids in index.list(prefix=f"{ds_id}#{file_id}#", namespace=namespace):
            log.info(f"OneDrive Pinecone Ids to delete: {ids}")
            index.delete(ids=ids, namespace=namespace)
        log.info(f"Removed Pinecone vectors with filter ds={ds_id}, file={file_id}")
    except Exception as e:
        log.exception(f"Error removing vector {vector_id_prefix} from Pinecone:")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception as e:
        log.exception(f"Error removing DB File {file_id}:")


def always_refresh_onedrive_token(ds):
    """
    Always refresh the OneDrive token using the refresh token, ignoring expiresAt logic.
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
        log.info(td)
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"[always_refresh_onedrive_token] No access_token in refresh response: {td}")
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
    Recursively list all files in OneDrive using MS Graph, ignoring subfolders or
    continuing if you want to recursively descend into subfolders.

    By default, OneDrive's root folder => folder_id="root" or "me/drive/root/children".
    If the user set ds["oneDriveFolderId"], we pass that folder_id in the Graph endpoint.
    """
    results = []
    stack = [folder_id]
    base_url = "https://graph.microsoft.com/v1.0/me/drive/items"  # or /me/drive/root if root
    headers = {"Authorization": f"Bearer {token}"}

    while stack:
        current_folder = stack.pop()
        # e.g. GET /me/drive/items/{folder_id}/children
        # We'll fetch them in pages with top=200 or so
        page_url = f"{base_url}/{current_folder}/children?$top=200"
        while page_url:
            log.debug(f"[_list_all_onedrive_files_recursive] Fetching => {page_url}")
            resp = requests.get(page_url, headers=headers)
            if not resp.ok:
                log.error(f"[_list_all_onedrive_files_recursive] List failed => {resp.status_code} - {resp.text[:500]}")
                if resp.status_code == 401:
                    raise ValueError("401 Unauthorized - Will attempt token refresh")
                resp.raise_for_status()

            data = resp.json()
            items = data.get("value", [])
            for item in items:
                # If item is a folder, push onto stack
                if item.get("folder"):
                    # If you want to recursively process subfolders, do stack.append(item["id"])
                    stack.append(item["id"])
                results.append(item)
            page_url = data.get("@odata.nextLink")

    return results


def list_all_onedrive_files_recursively_with_retry(access_token: str, folder_id: str, ds) -> List[dict]:
    """
    Similar to the Google Drive approach, we always refresh the OneDrive token first
    in the main logic. If we get a 401, we can re-raise so the caller can attempt refresh again.
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

    For standard files, we do GET /me/drive/items/{item_id}/content
    For Office docs, we can retrieve them as binary. If we want to export them
    to docx or PDF, we can do so via Graph conversion endpoints or local conversion.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/content"

    resp = requests.get(url, headers=headers, stream=True)
    if not resp.ok:
        log.error(f"[download_onedrive_file_content] Could not download => {resp.status_code} - {resp.text[:300]}")
        resp.raise_for_status()

    # Return raw content, let the local logic figure out MIME
    return resp.content


def convert_to_pdf(file_path, file_extension):
    """
    Same approach as your Google code: uses LibreOffice to convert to PDF.
    """
    try:
        pdf_file_path = os.path.splitext(file_path)[0] + ".pdf"
        command = [
            "/opt/libreoffice7.6/program/soffice",
            "--headless",
            "--convert-to",
            'pdf:writer_pdf_Export:{"SelectPdfVersion":{"type":"long","value":"17"}, "UseTaggedPDF": {"type":"boolean","value":"true"}}',
            "--outdir",
            os.path.dirname(file_path),
            file_path,
        ]
        subprocess.run(command, check=True)
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as f:
                pdf_data = f.read()
            os.remove(pdf_file_path)
            return pdf_data
        else:
            log.error("PDF file not found after conversion.")
            return None
    except subprocess.CalledProcessError as e:
        log.error(f"Conversion to PDF failed: {str(e)}")
        return None
    except Exception as e:
        log.error(f"Error during PDF conversion: {str(e)}")
        return None


async def _async_put_page(session, url, page_data, page_num, headers, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            payload_copy = io.BytesIO(page_data.getvalue())
            async with session.put(
                url,
                data=payload_copy,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                if resp.status == 429:
                    retries += 1
                    await asyncio.sleep(1)
                    continue
                content = await resp.read()
                text_content = content.decode("utf-8", errors="ignore")
                log.info(f"Page {page_num} => {len(text_content)} chars.")
                return text_content, page_num
        except (aiohttp.ClientError, ssl.SSLError, asyncio.TimeoutError) as e:
            log.warning(f"Error PUT page {page_num}: {str(e)}, retry {retries+1}/{max_retries}")
            retries += 1
            await asyncio.sleep(1)

    raise RuntimeError(f"Failed after {max_retries} tries for page {page_num}")


async def process_pages_async(
    pages, headers, filename, namespace, file_id, data_source_id, last_modified
):
    url = SERVER_URL
    log.info(f"Starting async processing of {len(pages)} pages w/ Tika at {url}.")
    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_put_page(session, url, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)

    log.info("All pages extracted. Now chunk + embed + upsert.")
    await create_and_upload_embeddings_in_batches(
        results, filename, namespace, file_id, data_source_id, last_modified
    )
    return results


def chunk_text(text, max_tokens=2048, overlap_chars=2000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    token_count = len(tokens)

    if token_count <= max_tokens:
        return [text]

    chunks = []
    start = 0
    end = max_tokens
    while start < token_count:
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)
        chunks.append(chunk_str)

        if end >= token_count:
            break

        overlap_str = (
            chunk_str[-overlap_chars:] if len(chunk_str) > overlap_chars else chunk_str
        )
        overlap_tokens = enc.encode(overlap_str)
        overlap_count = len(overlap_tokens)

        start = end - overlap_count
        end = start + max_tokens

    return chunks


async def create_and_upload_embeddings_in_batches(
    results, filename, namespace, file_id, data_source_id, last_modified
):
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    for text, page_num in results:
        if not text.strip():
            continue
        text_chunks = chunk_text(text, max_tokens=2048, overlap_chars=2000)
        for i, chunk in enumerate(text_chunks):
            chunk_tokens = enc.encode(chunk)
            chunk_token_len = len(chunk_tokens)

            # If adding this chunk exceeds limits, flush the batch
            if ((batch_text_count + 1) > max_batch_texts) or (
                batch_token_count + chunk_token_len > max_batch_tokens
            ):
                if batch:
                    await process_embedding_batch(
                        batch,
                        filename,
                        namespace,
                        file_id,
                        data_source_id,
                        last_modified,
                    )
                    batch.clear()
                    batch_token_count = 0
                    batch_text_count = 0

            if ((batch_text_count + 1) <= max_batch_texts) and (
                batch_token_count + chunk_token_len <= max_batch_tokens
            ):
                batch.append((chunk, page_num, i))
                batch_text_count += 1
                batch_token_count += chunk_token_len
            else:
                log.warning("Chunk too large or logic error preventing batch addition.")
                continue

    if batch:
        await process_embedding_batch(
            batch, filename, namespace, file_id, data_source_id, last_modified
        )
        batch.clear()


async def process_embedding_batch(
    batch, filename, namespace, file_id, data_source_id, last_modified
):
    texts = [item[0] for item in batch]
    embeddings = await get_google_embedding(texts)
    vectors = []
    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num,
            "sourceType": "oneDrive",
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            ),
        }
        vectors.append({"id": doc_id, "values": embedding, "metadata": metadata})

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)


async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    """
    If you're using Vertex AI for embeddings. Adjust if you have a separate approach for OneDrive.
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


def run_job():
    """
    Replicates the same logic as google_drive_ingest.py, but for OneDrive ingestion.
    """
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"Cloud Run Job: parsed event_data: {event_data}")
    except Exception as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    last_sync_time = event_data.get("lastSyncTime")
    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        log.info(f"data source from db: {ds}")
        if not ds:
            log.info(f"No DS found: {ds}")
            return (f"No DataSource for id={data_source_id}", 404)
        if ds["sourceType"] != "oneDrive":
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not oneDrive.", 400)

        # 1) Always refresh the token, ignoring any existing expiresAt.
        ds = always_refresh_onedrive_token(ds)
        access_token = ds.get("oneDriveAccessToken")
        log.info(f"refreshed access_token: {access_token}")
        if not access_token:
            return ("No valid oneDriveAccessToken after refresh", 400)

        folder_id = ds.get("oneDriveFolderId", "root")  # or "root" if top-level
        log.info(f"folder_id: {folder_id}")
        if not folder_id:
            return ("No oneDriveFolderId in DS record.", 400)

        if ds.get("lastSyncTime"):
            last_sync_time = ds["lastSyncTime"]
        log.info(f"last_sync_time: {last_sync_time}")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except Exception as e:
                log.error(f"Invalid last_sync_time format: {last_sync_time}, error: {e}")
                last_sync_dt = None
        log.info(f"last_sync_dt: {last_sync_dt}")
        # 2) List files from OneDrive
        try:
            od_files = list_all_onedrive_files_recursively_with_retry(access_token, folder_id, ds)
            log.info(f"od_files: {od_files}")
        except Exception as e:
            log.exception("Failed listing files from OneDrive even after token refresh:")
            psql.update_data_source_by_id(data_source_id, status="error")
            return (str(e), 500)

        log.info(f"Found {len(od_files)} total items in folder {folder_id} (including subfolders).")

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

        # OneDrive items typically have { "id", "name", "file", "folder", "lastModifiedDateTime", etc. }
        odFileId_to_key = {}
        od_file_keys = set()

        for item in od_files:
            item_id = item["id"]
            item_name = item.get("name", "untitled")
            item_file = item.get("file")
            item_folder = item.get("folder")
            item_modified_time = item.get("lastModifiedDateTime")  # e.g. "2023-09-01T12:03:43Z"

            # If it's a folder, we skip, because we've recursed deeper anyway
            if item_folder:
                continue

            # Generate unique file key
            file_key = generate_md5_hash(sub_for_hash, project_id, item_id)
            odFileId_to_key[item_id] = file_key
            od_file_keys.add(file_key)

            # Determine if new or updated
            if file_key not in db_file_keys:
                new_files.append(item)
            else:
                if last_sync_dt:
                    try:
                        item_modified_dt = ensure_timezone_aware(parser.isoparse(item_modified_time))
                    except Exception:
                        item_modified_dt = None

                    if item_modified_dt and item_modified_dt > last_sync_dt:
                        updated_files.append(item)
                    else:
                        log.info(f"No changes for item {item_name}.")
                else:
                    updated_files.append(item)

        removed_keys = db_file_keys - od_file_keys
        log.info(f"Removed keys: {removed_keys}")
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
                nextSyncTime=computeNextSyncTime(now_dt, ds.get("syncOption")).isoformat(),
            )
            return ("No new/updated files, done.", 200)

        if not SERVER_URL:
            log.error("SERVER_URL is not set. Cannot proceed with Tika processing.")
            psql.update_data_source_by_id(
                data_source_id,
                status="error",
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )
            return ("SERVER_URL is not set.", 500)

        # Some ID token logic if needed, or if your Tika server is internal
        # We'll assume your Tika server allows token-based or internal access
        # If you need a service account token => Implement similarly to google's approach

        # For consistency, let's assume we do not impersonate ID token for OneDrive scenario:
        headers = {
            "Authorization": f"Bearer some_token_or_internal_auth",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()

        db_file_keys_list = set(db_file_keys)
        for od_item in to_process:
            item_id = od_item["id"]
            item_name = od_item.get("name", "untitled")
            item_modified_time = od_item.get("lastModifiedDateTime")
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

            # 3) Download the file content from OneDrive
            try:
                content_bytes = download_onedrive_file_content(access_token, item_id)
            except Exception as e:
                log.exception(f"Failed to download {item_name} from OneDrive:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            if not content_bytes:
                log.error(f"Item {item_name} returned empty or invalid content.")
                psql.update_file_by_id(file_key, status="not supported", updatedAt=now_dt.isoformat())
                continue

            # We'll attempt local MIME detection or some extension guess
            # If you have actual MIME from item["file"]["mimeType"], use it
            item_mimeType = (od_item.get("file") or {}).get("mimeType", "application/octet-stream")

            # Save to a temp file
            ext = "pdf"  # or a logic to guess extension from MIME
            temp_file_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{ext}")
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing file {item_name} to disk:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            def process_local_file(temp_path, file_extension):
                """
                Same logic as your google drive approach:
                - If PDF => parse
                - If docx => convert to PDF => parse
                - If images => handle
                - If Excel => ...
                """
                if file_extension == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_extension in ["docx", "xlsx", "pptx", "odt", "etc..."]:
                    pdf_data = convert_to_pdf(temp_path, file_extension)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed")
                # else handle images, CSV, etc...
                # For brevity, same logic as your google code
                raise ValueError("Unsupported file format in OneDrive ingestion logic")

            try:
                final_pages, final_num_pages = process_local_file(temp_file_path, ext.lower())
            except Exception as e:
                log.error(f"Failed processing {item_name} as {ext}: {str(e)}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

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
                log.warning(f"Publish usage failed for file {item_name}: {str(e)}")

            del content_bytes, final_pages, results

        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=computeNextSyncTime(final_now_dt, ds.get("syncOption")).isoformat(),
        )
        return ("OK", 200)

    except Exception as e:
        log.exception("Error in oneDrive ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="error")
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
        log.error("SERVER_URL environment variable is not set correctly.")
        sys.exit(1)
    log.debug(f"SERVER_URL is set to: {SERVER_URL}")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
