#!/usr/bin/env python3
# google_drive_ingest.py

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
from shared import psql, google_pub_sub, google_auth, file_processor
from google.cloud import storage
from pinecone.grpc import PineconeGRPC as Pinecone

# If using Vertex AI for embeddings
from vertexai.preview.language_models import TextEmbeddingModel

# BM25 + sparse vector utilities (with no spaCy usage)
from shared.bm25 import (
    tokenize_document,
    publish_partial_bm25_update,
    compute_bm25_sparse_vector,
    get_project_vocab_stats
)

# ----------------------------------------------------------------------------
# ENV VARIABLES
# ----------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER", "/tmp/uploads")  # default

SERVER_DOMAIN = os.environ.get("SERVER_URL")  # e.g., mydomain.com
SERVER_URL = f"https://{SERVER_DOMAIN}/tika" if SERVER_DOMAIN else None

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")

# Define Central US Timezone
CENTRAL_TZ = ZoneInfo("America/Chicago")

# ----------------------------------------------------------------------------
# INITIALIZE PINECONE
# ----------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ----------------------------------------------------------------------------
# MIME TYPE -> EXTENSION MAP (for Drive export or alt=media)
# ----------------------------------------------------------------------------
GDRIVE_MIME_EXT_MAP = {
    "application/octet-stream": "use_extension",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/pdf": "pdf",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/vnd.oasis.opendocument.graphics": "odg",
    "application/vnd.oasis.opendocument.formula": "odf",
    "application/vnd.oasis.opendocument.flat.text": "fodt",
    "application/vnd.oasis.opendocument.flat.presentation": "fodp",
    "application/vnd.oasis.opendocument.flat.graphics": "fodg",
    "application/vnd.oasis.opendocument.spreadsheet-template": "ots",
    "application/vnd.oasis.opendocument.flat.spreadsheet-template": "fots",
    "application/vnd.lotus-1-2-3": "123",
    "application/dbase": "dbf",
    "text/html": "html",
    "application/vnd.lotus-screencam": "scm",
    "text/csv": "csv",
    "application/vnd.ms-excel": "xls",
    "application/vnd.ms-excel.template.macroenabled.12": "xltm",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template": "dotx",
    "application/vnd.ms-word.document.macroenabled.12": "docm",
    "application/vnd.ms-word.template.macroenabled.12": "dotm",
    "application/xml": "xml",
    "application/msword": "doc",
    "application/rtf": "rtf",
    "application/wordperfect": "wpd",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template": "xltx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsm",
    "application/vnd.corelqpw": "qpw",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow": "ppsx",
    "application/vnd.openxmlformats-officedocument.presentationml.slide": "ppmx",
    "application/vnd.openxmlformats-officedocument.presentationml.template": "potx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.12": "ppsm",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12": "pptm",
    "application/vnd.ms-powerpoint.addin.macroenabled.12": "ppam",
    "application/vnd.ms-powerpoint.slideshow": "pps",
    "application/vnd.ms-powerpoint.presentation": "ppt",
    "application/vnd.ms-powerpoint.addin": "ppa",
    "message/rfc822": "eml",
    "application/vnd.ms-outlook": "msg",
    "application/mbox": "mbox",
    "application/ost": "ost",
    "application/emlx": "emlx",
    "application/dbx": "dbx",
    "application/dat": "dat",
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
}

# ----------------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------------
def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int) -> dict:
    """
    - Combine all file text into one big string.
    - Publish partial update once so aggregator merges the entire file's token freq.
    - Fetch and return the updated vocab stats from Firestore.
    """
    # 1) Combine
    combined_text = " ".join(file_texts)

    # 2) Tokenize
    tokens = tokenize_document(combined_text)

    # 3) Publish partial update => aggregator merges term freq
    publish_partial_bm25_update(project_id, tokens, is_new_doc=True)

    # 4) Get BM25 stats from Firestore => { "N":..., "avgdl":..., "vocab": { term-> {df, tf} } }
    vocab_stats = get_project_vocab_stats(project_id)

    return compute_bm25_sparse_vector(tokens, project_id, vocab_stats, max_terms=max_terms)





def ensure_timezone_aware(dt):
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


def generate_md5_hash(*args):
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    md5_hash = hashlib.md5(combined_string.encode("utf-8")).hexdigest()
    return md5_hash


def computeNextSyncTime(from_date: datetime, sync_option: str = None) -> datetime:
    """
    Computes the next sync time based on the current date/time in Central US Time
    and syncOption. Options: 'hourly', 'daily', 'weekly', 'monthly'.
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
def always_refresh_drive_token(ds):
    """
    Always refresh the Google Drive token using the refresh token,
    regardless of whether the current one is valid or not.
    """
    refresh_token = ds.get("googleRefreshToken")
    if not refresh_token:
        log.warning(f"No refresh token for DS {ds['id']}; skipping refresh.")
        return ds

    # We no longer check ds["googleExpiresAt"]; always refresh
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
        log.info(td)
        new_access_token = td.get("access_token")
        if not new_access_token:
            log.error(f"[always_refresh_drive_token] No access_token in refresh response: {td}")
            return ds

        expires_in = td.get("expires_in", 3920)
        now_dt = datetime.now(CENTRAL_TZ)
        new_expires_dt = now_dt + timedelta(seconds=expires_in)
        new_expires_str = new_expires_dt.isoformat()
        psql.update_data_source_by_id(
            ds["id"],
            googleAccessToken=new_access_token,
            googleExpiresAt=new_expires_str,
        )
        ds["googleAccessToken"] = new_access_token
        ds["googleExpiresAt"] = new_expires_str
        log.info(f"[always_refresh_drive_token] Refreshed token for DS {ds['id']}, expiresAt={new_expires_dt}")
        return ds
    except Exception as e:
        log.exception("[always_refresh_drive_token] Error refreshing token for DS:")
        return ds


def _list_all_files_recursive(token: str, fld_id: str, ds) -> List[dict]:
    results = []
    stack = [fld_id]
    base_url = "https://www.googleapis.com/drive/v3/files"
    # For debugging: only log partial token
    token_str_for_log = (f"{token[:10]}...<redacted>...{token[-5:]}"
                         if len(token) > 15 else token)
    headers = {"Authorization": f"Bearer {token}"}
    fields = "files(id,name,mimeType,createdTime,modifiedTime,trashed),nextPageToken"

    log.debug(f"[_list_all_files_recursive] Using token => {token_str_for_log}")
    log.debug(f"[_list_all_files_recursive] Starting to list files in folder ID: {fld_id}")

    while stack:
        current_folder = stack.pop()
        query = f"'{current_folder}' in parents and trashed=false"
        page_token = None
        page_number = 1  # track which page within the folder

        log.debug(f"[_list_all_files_recursive] Processing folder ID: {current_folder}")

        while True:
            params = {
                "q": query,
                "fields": fields,
                "pageSize": 1000
            }
            if page_token:
                params["pageToken"] = page_token
                log.debug(f"[_list_all_files_recursive] Fetching page {page_number} for folder ID: {current_folder} "
                          f"with pageToken={page_token}")
            else:
                log.debug(f"[_list_all_files_recursive] Fetching first page for folder ID: {current_folder}")

            try:
                resp = requests.get(base_url, headers=headers, params=params)
            except requests.exceptions.RequestException as e:
                log.error(f"[_list_all_files_recursive] Request exception while listing files: {e}")
                raise

            log.debug(f"[_list_all_files_recursive] Response status code => {resp.status_code}")
            if not resp.ok:
                # Log up to 500 chars of response text for debugging
                resp_text_snippet = resp.text[:500]
                log.error(f"[_list_all_files_recursive] Failed to list files: {resp.status_code} - {resp_text_snippet}")
                if resp.status_code == 401:
                    # We will let the caller handle refresh logic
                    log.warning("[_list_all_files_recursive] Received 401 Unauthorized => raising for refresh logic.")
                    raise ValueError("401 Unauthorized - Will attempt token refresh")
                resp.raise_for_status()

            data = resp.json()
            files = data.get("files", [])
            log.debug(f"[_list_all_files_recursive] Fetched {len(files)} files in page {page_number} for folder={current_folder}")

            for f in files:
                results.append(f)
                if f.get("mimeType") == "application/vnd.google-apps.folder":
                    log.debug(f"[_list_all_files_recursive] Found subfolder: {f.get('name')} (ID: {f.get('id')}) => pushing to stack.")
                    stack.append(f["id"])

            page_token = data.get("nextPageToken")
            if not page_token:
                log.debug(f"[_list_all_files_recursive] No more pages for folder={current_folder}")
                break
            else:
                page_number += 1

    log.debug(f"[_list_all_files_recursive] Total files fetched => {len(results)}")
    return results


def remove_file_from_db_and_pinecone(file_id, ds_id, project_id, namespace):
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        # index.delete(filter={"dataSourceId": ds_id, "fileId": file_id}, namespace=namespace)
        for ids in index.list(prefix=f"{ds_id}#{file_id}#", namespace="example-namespace"):
            log.info(f"Pinecone Ids to delete: {ids}")
            index.delete(ids=ids, namespace=namespace)
        log.info(f"Removed Pinecone vectors with filter ds={ds_id}, file={file_id}")
    except Exception as e:
        log.exception(f"Error removing vector {vector_id_prefix} from Pinecone:")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception as e:
        log.exception(f"Error removing DB File {file_id}:")



def list_all_files_recursively_with_retry(access_token: str, folder_id: str, ds) -> List[dict]:
    """
    Recursively lists files (including subfolders) in the given Drive folder.
    Because we always refresh the token beforehand, we skip the 401 retry logic.
    """
    log.debug("[list_all_files_recursively_with_retry] Starting. "
              f"Folder ID={folder_id}, token (truncated) => {access_token[:10]}...{access_token[-5:]}")
    try:
        log.info("[list_all_files_recursively_with_retry] Listing files with freshly obtained token.")
        drive_files = _list_all_files_recursive(access_token, folder_id, ds)
        return drive_files
    except Exception as e:
        log.error(f"[list_all_files_recursively_with_retry] Failed to list files => {e}")
        raise



def download_drive_file_content(access_token, file_id, mime_type=None):
    if mime_type and mime_type.startswith("application/vnd.google-apps."):
        export_ext = GDRIVE_MIME_EXT_MAP.get(mime_type, "pdf")
        if export_ext == "docx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif export_ext == "xlsx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif export_ext == "pptx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.presentationml.presentation"
        else:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/pdf"
            export_ext = "pdf"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        return r.content, export_ext
    else:
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        ext = GDRIVE_MIME_EXT_MAP.get(mime_type, "NA") if mime_type else "NA"
        return r.content, ext


def convert_to_pdf(file_path, file_extension):
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
            log.warning(
                f"Error PUT page {page_num}: {str(e)}, retry {retries+1}/{max_retries}"
            )
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
    # 1) # ADDED for single-file BM25 update
    all_file_texts = [text for (text, _page_num) in results if text.strip()]
    # Update aggregator once, then retrieve vocab_stats
    sparse_values = get_sparse_vector(all_file_texts, namespace, 300)
    for text, page_num in results:
        if not text.strip():
            continue
        text_chunks = chunk_text(text, max_tokens=2048, overlap_chars=2000)
        for i, chunk in enumerate(text_chunks):
            chunk_tokens = enc.encode(chunk)
            chunk_token_len = len(chunk_tokens)

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
                        sparse_values
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
            batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
        )
        batch.clear()


async def process_embedding_batch(
    batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
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
            "sourceType": "google drive",
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            ),
        }
        vectors.append({"id": doc_id, "values": embedding,"sparse_values":sparse_values, "metadata": metadata})

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)


async def get_google_embedding(
    queries, model_name="text-multilingual-embedding-preview-0409"
):
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


def run_job():
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
        if not ds:
            return (f"No DataSource for id={data_source_id}", 404)
        if ds["sourceType"] != "googleDrive":
            return (
                f"DataSource {data_source_id} is {ds['sourceType']}, not googleDrive.",
                400,
            )

        # 1) Always refresh the token here, ignoring any existing token
        ds = always_refresh_drive_token(ds)
        access_token = ds.get("googleAccessToken")
        if not access_token:
            return ("No valid googleAccessToken after refresh", 400)

        folder_id = ds.get("googleDriveFolderId")
        if not folder_id:
            return ("No googleDriveFolderId in DS record.", 400)

        if ds.get("lastSyncTime"):
            last_sync_time = ds["lastSyncTime"]

        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except Exception as e:
                log.error(
                    f"Invalid last_sync_time format: {last_sync_time}, error: {e}"
                )
                last_sync_dt = None

        # Attempt to list files from Google Drive with possible 401 retry
        try:
            drive_files = list_all_files_recursively_with_retry(access_token, folder_id, ds)
        except Exception as e:
            log.exception("Failed listing files from Google Drive even after retry:")
            psql.update_data_source_by_id(data_source_id, status="error")
            return (str(e), 500)

        log.info(
            f"Found {len(drive_files)} total files (including subfolders) in folder {folder_id}."
        )

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

        driveFileId_to_key = {}
        drive_file_keys = set()
        for gf in drive_files:
            gf_id = gf["id"]
            gf_name = gf["name"]
            gf_mime = gf.get("mimeType")
            gf_created_time = gf.get("createdTime")
            gf_modified_time = gf.get("modifiedTime")

            if gf_mime and "folder" in gf_mime:
                log.info(f"Skipping subfolder {gf_name}, but continuing recursion.")
                continue

            gf_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gf_id)
            driveFileId_to_key[gf_id] = gf_key
            drive_file_keys.add(gf_key)

            if gf_key not in db_file_keys:
                new_files.append(gf)
            else:
                if last_sync_dt:
                    try:
                        gf_created_dt = (
                            ensure_timezone_aware(parser.isoparse(gf_created_time))
                            if gf_created_time
                            else None
                        )
                    except Exception:
                        gf_created_dt = None
                    try:
                        gf_modified_dt = (
                            ensure_timezone_aware(parser.isoparse(gf_modified_time))
                            if gf_modified_time
                            else None
                        )
                    except Exception:
                        gf_modified_dt = None

                    if (gf_created_dt and gf_created_dt > last_sync_dt) or (
                        gf_modified_dt and gf_modified_dt > last_sync_dt
                    ):
                        updated_files.append(gf)
                    else:
                        log.info(f"No changes for file {gf_name}.")
                else:
                    updated_files.append(gf)

        removed_keys = db_file_keys - drive_file_keys
        log.info(f"Removed keys: {removed_keys}")
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(
                r_key, data_source_id, project_id, namespace=project_id
            )

        log.info(f"New files => {len(new_files)}, Updated => {len(updated_files)}")
        to_process = new_files + updated_files
        if not to_process:
            log.info("No new or updated files => done.")
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

        try:
            bearer_token = google_auth.impersonated_id_token(serverurl=SERVER_DOMAIN).json()["token"]
        except Exception as e:
            log.exception("Failed to obtain impersonated ID token:")
            psql.update_data_source_by_id(
                data_source_id,
                status="error",
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )
            return ("Failed to obtain impersonated ID token.", 500)

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()

        db_file_keys_list = set(db_file_keys)
        for gf in to_process:
            gf_id = gf["id"]
            gf_name = gf["name"]
            gf_mime = gf.get("mimeType", "application/octet-stream")
            gf_key = driveFileId_to_key[gf_id]

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

            if not gf_mime or (
                gf_mime not in GDRIVE_MIME_EXT_MAP
                and not gf_mime.startswith("application/vnd.google-apps.")
            ):
                log.error(f"MIME type {gf_mime} is not supported for {gf_name}.")
                psql.update_file_by_id(gf_key, status="not supported", updatedAt=now_dt.isoformat())
                continue

            try:
                content_bytes, export_ext = download_drive_file_content(access_token, gf_id, mime_type=gf_mime)
                if export_ext == "NA":
                    continue
            except Exception as e:
                log.exception(f"Failed to download {gf_name} from Drive:")
                psql.update_file_by_id(gf_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            if content_bytes is None or export_ext is None:
                log.error(f"File {gf_name} has an unsupported MIME type or export failed.")
                psql.update_file_by_id(gf_key, status="not supported", updatedAt=now_dt.isoformat())
                continue

            temp_file_path = os.path.join(UPLOADS_FOLDER, f"{gf_key}.{export_ext}")
            try:
                with open(temp_file_path, "wb") as f:
                    f.write(content_bytes)
            except Exception as e:
                log.exception(f"Error writing file {gf_name} to disk:")
                psql.update_file_by_id(gf_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            def process_local_file(temp_path, file_extension):
                if file_extension == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
                    with open(temp_path, "rb") as f_in:
                        excel_data = f_in.read()
                    pages_local = file_processor.split_excel(excel_data)
                    return (pages_local, len(pages_local))
                elif file_extension in ["eml", "msg", "pst", "ost", "mbox", "dbx", "dat", "emlx"]:
                    with open(temp_path, "rb") as f_in:
                        email_data = f_in.read()
                    return ([("1", io.BytesIO(email_data))], 1)
                elif file_extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
                    with open(temp_path, "rb") as f_in:
                        image_data = f_in.read()
                    return ([("1", io.BytesIO(image_data))], 1)
                elif file_extension == "ods":
                    with open(temp_path, "rb") as f_in:
                        ods_data = f_in.read()
                    pages_local = file_processor.split_ods(ods_data)
                    return (pages_local, len(pages_local))
                elif file_extension in [
                    "docx",
                    "pdf",
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
                    pdf_data = convert_to_pdf(temp_path, file_extension)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed")
                else:
                    raise ValueError("Unsupported file format")

            try:
                final_pages, final_num_pages = process_local_file(temp_file_path, export_ext.lower())
            except Exception as e:
                log.error(f"Failed processing {gf_name} as {export_ext}: {str(e)}")
                psql.update_file_by_id(gf_key, status="failed", updatedAt=now_dt.isoformat())
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
                        gf_name,
                        project_id,
                        gf_key,
                        data_source_id,
                        last_modified=now_dt,
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding step for {gf_name}:")
                psql.update_file_by_id(gf_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            psql.update_file_by_id(
                gf_key,
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
                log.warning(f"Publish usage failed for file {gf_name}: {str(e)}")

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
        log.exception("Error in googleDrive ingestion flow:")
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
