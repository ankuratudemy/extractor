#!/usr/bin/env python3
# sharepoint_ingest.py

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
from typing import List, Tuple, Optional, Dict
from zoneinfo import ZoneInfo

from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, file_processor  # Adjust if needed
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

SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET")

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

def ensure_timezone_aware(dt: datetime) -> datetime:
    """
    Ensures that the datetime object is timezone-aware.
    If it's naive, assume Central US Time and make it aware.
    """
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


def generate_md5_hash(*args) -> str:
    """
    Generates an MD5 hash from the JSON-dumped data
    ensuring uniqueness for user-specific combos.
    """
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    return hashlib.md5(combined_string.encode("utf-8")).hexdigest()


def computeNextSyncTime(from_date: datetime, sync_option: str = None) -> datetime:
    """
    Computes the next sync time based on the current date/time in Central US Time
    and sync_option. Options: 'hourly', 'daily', 'weekly', 'monthly'.
    Fallback to 1 hour if no valid sync_option is provided.
    """
    if not isinstance(from_date, datetime):
        from_date = datetime.now(CENTRAL_TZ)

    if not sync_option:
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


def remove_file_from_db_and_pinecone(file_id: str, ds_id: str, project_id: str, namespace: str):
    """
    Removes references in DB + Pinecone for the given file_id => consistent with GoogleDrive logic.
    """
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        # Example: We'll assume a prefix-based list for Pinecone
        for ids in index.list(prefix=f"{ds_id}#{file_id}#", namespace=namespace):
            log.info(f"SharePoint Pinecone Ids to delete: {ids}")
            index.delete(ids=ids, namespace=namespace)
        log.info(f"Removed Pinecone vectors with filter ds={ds_id}, file={file_id}")
    except Exception as e:
        log.exception(f"Error removing vector {vector_id_prefix} from Pinecone:")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception as e:
        log.exception(f"Error removing DB File {file_id}:")


def always_refresh_sharepoint_token(ds: Dict) -> Dict:
    """
    Always refresh the SharePoint token using the refresh token, ignoring expiresAt logic.
    """
    refresh_token = ds.get("sharePointRefreshToken")
    if not refresh_token:
        log.warning(f"No sharePointRefreshToken for DS {ds['id']}; skipping refresh.")
        return ds

    token_url = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
    body_params = {
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": "https://graph.microsoft.com/.default offline_access"
    }
    try:
        resp = requests.post(token_url, data=body_params)
        if not resp.ok:
            log.error(f"[always_refresh_sharepoint_token] Refresh request failed: {resp.text}")
            return ds
        td = resp.json()
        log.info(td)
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"[always_refresh_sharepoint_token] No access_token in refresh response: {td}")
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
        log.info(f"[always_refresh_sharepoint_token] Refreshed token for DS {ds['id']}, expiresAt={new_expires_dt}")
        return ds
    except Exception as e:
        log.exception("[always_refresh_sharepoint_token] Error refreshing token for DS:")
        return ds


# ----------------------------------------------------------------------------
# FALLBACK LIST LOGIC
# ----------------------------------------------------------------------------
def _list_site_based(token: str, site_id: str, folder_id: str) -> List[dict]:
    """
    Uses /sites/{siteId}/drive/items/{folder_id}/children to list items.
    This approach works if 'folder_id' is actually an item in the default
    doc library for the given site. If it fails, we fallback.
    """
    base_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{folder_id}/children?$top=200"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    stack = [base_url]
    while stack:
        url = stack.pop()
        log.debug(f"[_list_site_based] GET => {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # might raise HTTPError(400)
        data = resp.json()
        items = data.get("value", [])
        for item in items:
            # If it's a folder, push to stack
            if "folder" in item:
                folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{item['id']}/children?$top=200"
                stack.append(folder_url)
            results.append(item)
        if "@odata.nextLink" in data:
            stack.append(data["@odata.nextLink"])
    return results


def _list_drive_based(token: str, drive_id: str, folder_id: Optional[str] = None) -> List[dict]:
    """
    Uses /drives/{driveId}/items/{folderId or root}/children to list items.
    If folder_id is None, defaults to 'root'.
    """
    if not folder_id:
        folder_id = "root"
    base_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children?$top=200"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    stack = [base_url]
    while stack:
        url = stack.pop()
        log.debug(f"[_list_drive_based] GET => {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # might raise HTTPError(400)
        data = resp.json()
        items = data.get("value", [])
        for item in items:
            if "folder" in item:
                sub_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item['id']}/children?$top=200"
                stack.append(sub_url)
            results.append(item)
        if "@odata.nextLink" in data:
            stack.append(data["@odata.nextLink"])
    return results


def _list_spo_list_items(token: str, site_id: str, list_id: str) -> List[dict]:
    """
    If it's truly a SharePoint List, uses /sites/{siteId}/lists/{listId}/items?$expand=fields
    to list items. No fallback needed for 'lists' if you are sure it's a list.
    """
    base_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?$expand=fields"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    next_url = base_url
    while next_url:
        log.debug(f"[_list_spo_list_items] GET => {next_url}")
        resp = requests.get(next_url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("value", [])
        results.extend(items)
        next_url = data.get("@odata.nextLink")
    return results


def list_all_sharepoint_files_fallback(token: str, site_id: str, list_or_folder_id: str, is_list: bool) -> List[dict]:
    """
    If is_list is True, we call the normal SharePoint list items endpoint.
    Otherwise, we first try the site-based approach: /sites/{siteId}/drive/items/{folderId}/children.
    If that fails with a 400 or 404, we fallback to drive-based approach: /drives/{list_or_folder_id}/items/root/children.
    """
    log.debug("[list_all_sharepoint_files_fallback] Starting.")
    if is_list:
        # It's a real SharePoint list => just use the list endpoint
        return _list_spo_list_items(token, site_id, list_or_folder_id)

    # Not a list => let's try site-based approach first
    try:
        log.info("[list_all_sharepoint_files_fallback] Trying site-based approach...")
        return _list_site_based(token, site_id, list_or_folder_id)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code in (400, 404):
            # fallback
            log.warning("[list_all_sharepoint_files_fallback] Site-based approach failed. Trying drive-based approach...")
            # In a drive-based approach, 'list_or_folder_id' is typically the 'driveId'
            # We'll assume we want to list 'root' or 'folder' items from that drive
            return _list_drive_based(token, list_or_folder_id, None)
        else:
            raise e  # re-raise if not a typical fallback scenario


# ----------------------------------------------------------------------------
# FALLBACK DOWNLOAD LOGIC
# ----------------------------------------------------------------------------
def _download_site_based(token: str, site_id: str, item_id: str) -> bytes:
    """
    Site-based: GET /sites/{siteId}/drive/items/{itemId}/content
    """
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    return resp.content


def _download_drive_based(token: str, drive_id: str, item_id: str) -> bytes:
    """
    Drive-based: GET /drives/{driveId}/items/{item_id}/content
    """
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    return resp.content


def download_file_with_fallback(token: str, site_id: str, item_id: str, maybe_drive_id: str) -> bytes:
    """
    Tries site-based approach first. If that fails with 400/404, fallback to drive-based approach.
    'maybe_drive_id' can be your 'list_or_folder_id' if it's actually a drive ID.
    """
    # First try site-based
    try:
        log.info("[download_file_with_fallback] Trying site-based download first...")
        return _download_site_based(token, site_id, item_id)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code in (400, 404):
            log.warning("[download_file_with_fallback] Site-based download failed => fallback to drive-based.")
            return _download_drive_based(token, maybe_drive_id, item_id)
        else:
            raise e


# ----------------------------------------------------------------------------
# OTHER PROCESSING LOGIC (Tika, Pinecone, etc.)
# ----------------------------------------------------------------------------

def convert_to_pdf(file_path: str, file_extension: str) -> Optional[bytes]:
    """
    Converts a file to PDF using LibreOffice.
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
    """
    Asynchronously uploads a page to Tika.
    """
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
    pages: List[Tuple[str, io.BytesIO]],
    headers: Dict,
    filename: str,
    namespace: str,
    file_id: str,
    data_source_id: str,
    last_modified: datetime
):
    """
    Asynchronously processes pages: uploads to Tika, creates embeddings, and uploads to Pinecone.
    """
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


def chunk_text(text: str, max_tokens: int = 2048, overlap_chars: int = 2000) -> List[str]:
    """
    Chunks text into smaller segments based on token count and character overlap.
    """
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
    results: List[Tuple[str, int]],
    filename: str,
    namespace: str,
    file_id: str,
    data_source_id: str,
    last_modified: datetime
):
    """
    Creates embeddings for chunks of text and uploads them to Pinecone in batches.
    """
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
    batch: List[Tuple[str, int, int]],
    filename: str,
    namespace: str,
    file_id: str,
    data_source_id: str,
    last_modified: datetime
):
    """
    Processes a batch of text chunks: creates embeddings and upserts them to Pinecone.
    """
    texts = [item[0] for item in batch]
    embeddings = await get_sharepoint_embedding(texts)
    vectors = []
    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num,
            "sourceType": "sharepoint",
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


async def get_sharepoint_embedding(
    queries: List[str], model_name: str = "text-multilingual-embedding-preview-0409"
) -> List[List[float]]:
    """
    Retrieves embeddings for a list of queries using Vertex AI.
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


# ----------------------------------------------------------------------------
# MAIN RUN FUNCTION
# ----------------------------------------------------------------------------

def run_job():
    """
    Main function to run the SharePoint ingestion job.
    """
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit("Function execution failed.")

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"Cloud Run Job: parsed event_data: {event_data}")
    except json.JSONDecodeError as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit("Function execution failed due to JSON parsing error.")

    last_sync_time = event_data.get("lastSyncTime")
    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        if ds is None:
            log.error(f"No DataSource found for id: {data_source_id}")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("Data source not found.")

        if ds.get("sourceType") != "sharepoint":
            log.error(f"DataSource {data_source_id} is {ds.get('sourceType')}, not valid.")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("Invalid data source type.")

        # 1) Always refresh the SharePoint token
        ds = always_refresh_sharepoint_token(ds)
        access_token = ds.get("sharePointAccessToken")
        if not access_token:
            log.error("No valid sharePointAccessToken after refresh.")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("No valid sharePointAccessToken after refresh.")

        site_id = ds.get("sharePointSiteId")
        list_id = ds.get("sharePointListId")
        folder_id = ds.get("sharePointFolderId")

        if not site_id:
            log.error("No sharePointSiteId in DS record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("No sharePointSiteId in DS record.")

        if not (list_id or folder_id):
            log.error("No sharePointListId or sharePointFolderId in DS record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("No sharePointListId or sharePointFolderId in DS record.")

        # If ds has an explicit lastSyncTime
        if ds.get("lastSyncTime"):
            last_sync_time = ds["lastSyncTime"]

        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except Exception as e:
                log.error(f"Invalid last_sync_time format: {last_sync_time}, error: {e}")
                last_sync_dt = None

        # 2) List files using our fallback approach
        is_list = bool(list_id)
        list_or_folder_id = list_id if is_list else folder_id

        try:
            sharepoint_files = list_all_sharepoint_files_fallback(
                access_token, site_id, list_or_folder_id, is_list
            )
        except Exception as e:
            log.exception("Failed listing files from SharePoint (both approaches):")
            psql.update_data_source_by_id(data_source_id, status="error")
            sys.exit("Failed listing files from SharePoint.")

        log.info(f"Found {len(sharepoint_files)} total items in {'list' if is_list else 'folder'} {list_or_folder_id}.")

        # 3) Identify new/updated/removed
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

        sharepointFileId_to_key = {}
        sharepoint_file_keys = set()

        for item in sharepoint_files:
            item_id = item["id"]     # file/folder item ID
            item_name = item.get("name", "untitled")
            item_folder = item.get("folder")
            item_modified_time = item.get("lastModifiedDateTime")

            # Skip folders since we added them to stack
            if item_folder:
                continue

            file_key = generate_md5_hash(sub_for_hash, project_id, item_id)
            sharepointFileId_to_key[item_id] = file_key
            sharepoint_file_keys.add(file_key)

            # Determine if new or updated
            if file_key not in db_file_keys:
                new_files.append(item)
            else:
                if last_sync_dt and item_modified_time:
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

        removed_keys = db_file_keys - sharepoint_file_keys
        log.info(f"Removed keys: {removed_keys}")
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(
                r_key, data_source_id, project_id, namespace=project_id
            )

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
            sys.exit("SERVER_URL is not set.")

        headers = {
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()

        # 4) Process each file
        db_file_keys_list = set(db_file_keys)
        for sp_item in to_process:
            item_id = sp_item["id"]
            item_name = sp_item.get("name", "untitled")
            file_key = sharepointFileId_to_key[item_id]
            now_dt = datetime.now(CENTRAL_TZ)

            # Add or update DB row
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
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            # Download file with fallback:
            try:
                content_bytes = download_file_with_fallback(access_token, site_id, item_id, list_or_folder_id)
            except Exception as e:
                log.exception(f"Failed to download {item_name} from SharePoint (both approaches):")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            if not content_bytes:
                log.error(f"Item {item_name} returned empty or invalid content.")
                psql.update_file_by_id(file_key, status="not supported", updatedAt=now_dt.isoformat())
                continue

            # Decide file extension
            filename_parts = item_name.split(".")
            if len(filename_parts) > 1:
                ext = filename_parts[-1].lower()
            else:
                ext = "pdf"  # default

            temp_file_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{ext}")
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing file {item_name} to disk:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            # Convert & split
            def process_local_file(path_: str, file_ext: str) -> Tuple[List[Tuple[str, io.BytesIO]], int]:
                if file_ext == "pdf":
                    with open(path_, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_ext in ["docx", "xlsx", "pptx", "odt", "odp", "ods"]:
                    pdf_data = convert_to_pdf(path_, file_ext)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed.")
                elif file_ext in ["csv", "tsv"]:
                    with open(path_, "rb") as f_in:
                        csv_data = f_in.read()
                    return file_processor.split_csv(csv_data)
                elif file_ext in ["eml", "msg", "pst", "ost", "mbox", "dbx", "dat", "emlx"]:
                    with open(path_, "rb") as f_in:
                        email_data = f_in.read()
                    return ([("1", io.BytesIO(email_data))], 1)
                elif file_ext in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
                    with open(path_, "rb") as f_in:
                        image_data = f_in.read()
                    return ([("1", io.BytesIO(image_data))], 1)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")

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

            # Send pages to Tika & embed
            try:
                results = loop.run_until_complete(
                    process_pages_async(
                        final_pages,
                        headers,
                        item_name,
                        project_id,  # namespace
                        file_key,    # file_id
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
        log.exception("Error in SharePoint ingestion flow:")
        psql.update_data_source_by_id(data_source_id, status="error")
        sys.exit("Function execution failed due to an unexpected error.")


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
        sys.exit("Function execution failed.")

    log.debug(f"SERVER_URL is set to: {SERVER_URL}")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
