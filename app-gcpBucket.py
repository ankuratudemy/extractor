#!/usr/bin/env python3
# gcp_ingest.py

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
from datetime import datetime, timedelta
from typing import List, Tuple
from zoneinfo import ZoneInfo

from dateutil import parser

# Logging + Shared Imports
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
from pinecone.grpc import PineconeGRPC as Pinecone

# If using Vertex AI for embeddings:
from vertexai.preview.language_models import TextEmbeddingModel

# GCP storage
from google.cloud import storage
from google.oauth2 import service_account

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
CENTRAL_TZ = ZoneInfo("America/Chicago")



SERVER_DOMAIN = os.environ.get("SERVER_URL")
SERVER_URL = f"https://{SERVER_DOMAIN}/tika" if SERVER_DOMAIN else None

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# -----------------------------------------------------------------------------------
# INITIALIZE PINECONE
# -----------------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# -----------------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------------
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
    """Ensure a naive datetime is assigned CENTRAL_TZ if tzinfo is missing."""
    if dt and dt.tzinfo is None:
        return dt.replace(tzinfo=CENTRAL_TZ)
    return dt

def get_event_loop():
    """Get or create the asyncio event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def generate_md5_hash(*args):
    """
    Generate an MD5 hash from a combination of JSON-serialized args.
    Used to create unique File IDs or keys.
    """
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    return hashlib.md5(combined_string.encode("utf-8")).hexdigest()

def compute_next_sync_time(from_date: datetime, sync_option: str = None) -> datetime:
    """Compute next sync time based on 'hourly', 'daily', 'weekly', 'monthly'."""
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

def remove_file_from_db_and_pinecone(file_id, ds_id, project_id, namespace):
    """
    Remove a file record from the DB and associated Pinecone vectors
    by prefixing with ds_id#file_id#.
    """
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        pinecone_ids = index.list(prefix=vector_id_prefix, namespace=namespace)
        if pinecone_ids:
            log.info(f"Found {len(pinecone_ids)} Pinecone IDs to delete for prefix={vector_id_prefix}")
            index.delete(ids=pinecone_ids, namespace=namespace)
            log.info(f"Removed Pinecone vectors with prefix={vector_id_prefix}")
        else:
            log.info(f"No Pinecone vectors found with prefix={vector_id_prefix}")
    except Exception as e:
        log.error(f"Error removing vectors with prefix {vector_id_prefix} from Pinecone: {e}")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception as e:
        log.error(f"Error removing DB File {file_id}: {e}")

def convert_to_pdf(file_path, file_extension):
    """
    Convert a non-PDF file (docx, xlsx, pptx, etc.) to PDF using LibreOffice headless.
    """
    try:
        pdf_file_path = os.path.splitext(file_path)[0] + ".pdf"
        command = [
            "/opt/libreoffice7.6/program/soffice",
            "--headless",
            "--convert-to",
            'pdf:writer_pdf_Export:{"SelectPdfVersion":{"type":"long","value":"17"}}',
            "--outdir",
            os.path.dirname(file_path),
            file_path,
        ]
        subprocess.run(command, check=True)
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, "rb") as f:
                pdf_data = f.read()
            os.remove(pdf_file_path)
            log.info(f"Successfully converted {file_path} to PDF.")
            return pdf_data
        else:
            log.error(f"PDF file not found after conversion: {pdf_file_path}")
            return None
    except subprocess.CalledProcessError as e:
        log.error(f"Conversion to PDF failed for {file_path}: {str(e)}")
        return None
    except Exception as e:
        log.error(f"Error during PDF conversion for {file_path}: {str(e)}")
        return None

async def _async_put_page(session, url, page_data, page_num, headers, max_retries=10):
    """
    PUT a single page to Tika for OCR/processing, with retry logic for 429 or 5xx errors.
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
                    log.warning(f"Rate limited on page {page_num}. Retry {retries}/{max_retries}")
                    await asyncio.sleep(1)
                    continue
                elif resp.status >= 500:
                    retries += 1
                    log.warning(f"Server error {resp.status} on page {page_num}. Retry {retries}/{max_retries}")
                    await asyncio.sleep(1)
                    continue
                elif resp.status != 200:
                    log.error(f"Unexpected status {resp.status} on page {page_num}; no further retries.")
                    return None, page_num

                content = await resp.read()
                text_content = content.decode("utf-8", errors="ignore")
                log.info(f"Successfully processed page {page_num}: {len(text_content)} characters.")
                return text_content, page_num

        except (aiohttp.ClientError, ssl.SSLError, asyncio.TimeoutError) as e:
            retries += 1
            log.warning(f"Error PUT page {page_num}: {str(e)}; retry {retries}/{max_retries}")
            await asyncio.sleep(1)

    log.error(f"Failed after {max_retries} retries for page {page_num}.")
    return None, page_num

async def process_pages_async(pages, headers, filename, namespace, file_id, data_source_id, last_modified):
    """
    Run Tika processing asynchronously for all pages, gather results, then embed + upsert to Pinecone.
    """
    if not SERVER_URL:
        raise ValueError("SERVER_URL is not set, cannot call Tika.")
    url = SERVER_URL
    log.info(f"Starting async Tika processing of {len(pages)} pages.")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_put_page(session, url, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)  # Wait for all tasks

    # Filter out failed pages
    successful_results = [res for res in results if res[0] is not None]
    failed_pages = [res[1] for res in results if res[0] is None]
    if failed_pages:
        log.error(f"Failed to process {len(failed_pages)} pages: {failed_pages}")

    log.info("All pages extracted. Now chunk, embed, and upsert to Pinecone.")
    await create_and_upload_embeddings_in_batches(
        successful_results, filename, namespace, file_id, data_source_id, last_modified
    )
    return successful_results

def chunk_text(text, max_tokens=2048, overlap_chars=2000):
    """
    Split text into overlapping chunks based on a max token count (2048 by default).
    Uses tiktoken with the 'cl100k_base' encoding.
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

        # Overlap to preserve context
        overlap_str = chunk_str[-overlap_chars:] if len(chunk_str) > overlap_chars else chunk_str
        overlap_tokens = enc.encode(overlap_str)
        overlap_count = len(overlap_tokens)

        start = end - overlap_count
        end = start + max_tokens
    return chunks

async def create_and_upload_embeddings_in_batches(
    results, filename, namespace, file_id, data_source_id, last_modified
):
    """
    1) Gather all text for the entire file.
    2) Update aggregator (once) for the entire file, fetch vocab_stats.
    3) Then chunk, embed, and compute sparse vectors for each chunk using that single vocab_stats.
    """
    # 1) # ADDED for single-file BM25 update
    all_file_texts = [text for (text, _page_num) in results if text.strip()]
    # Update aggregator once, then retrieve vocab_stats
    sparse_values = get_sparse_vector(all_file_texts, namespace, 300)

    # The rest is basically the same, but we pass vocab_stats along
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    total_chunks = 0
    for text, page_num in results:
        if not text.strip():
            continue

        text_chunks = chunk_text(text, max_tokens=2048, overlap_chars=2000)
        for i, chunk in enumerate(text_chunks):
            chunk_token_len = len(enc.encode(chunk))
            # If adding this chunk exceeds any limit, process the current batch
            if (batch_text_count + 1) > max_batch_texts or (batch_token_count + chunk_token_len > max_batch_tokens):
                if batch:
                    await process_embedding_batch(
                        batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
                    )
                    total_chunks += len(batch)
                    batch.clear()
                    batch_text_count = 0
                    batch_token_count = 0

            if (batch_text_count + 1) <= max_batch_texts and (batch_token_count + chunk_token_len <= max_batch_tokens):
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
        total_chunks += len(batch)
        batch.clear()

    log.info(f"Total text chunks processed and upserted: {total_chunks}")

async def process_embedding_batch(
    batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
):
    texts = [item[0] for item in batch]
    embeddings = await get_google_embedding(texts)
    vectors = []

    for (chunk_text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        # Replaced old “get_sparse_values” with the new helper that uses pre-fetched stats

        doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": chunk_text,
            "source": filename,
            "page": page_num,
            "sourceType": "gcpBucket",
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified),
        }
        vectors.append({
            "id": doc_id,
            "values": embedding,
            "sparse_values": sparse_values,
            "metadata": metadata
        })

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        log.info(f"Successfully upserted {len(vectors)} vectors to Pinecone.")
    except Exception as e:
        log.error(f"Failed to upsert vectors to Pinecone for file_id={file_id}: {e}")

async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    """
    Use Vertex AI to get embeddings for a list of queries (chunks).
    """
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
        embeddings = [emb.values for emb in embeddings_list]
        log.info(f"Successfully obtained embeddings for {len(embeddings)} texts.")
        return embeddings
    except Exception as e:
        log.error(f"Failed to get embeddings from Vertex AI: {e}")
        raise

def connect_gcp_storage(credential_json):
    """
    Connect to GCP Storage using the provided service account JSON.
    """
    try:
        info = json.loads(credential_json)
        creds = service_account.Credentials.from_service_account_info(info)
        client = storage.Client(credentials=creds)
        log.info("Successfully connected to GCP Storage.")
        return client
    except Exception as e:
        log.error("Failed to parse or use GCP credentials.")
        raise

def run_job():
    """
    Main entrypoint for GCP Bucket ingestion. 
    Pulls from DATA_SOURCE_CONFIG env var -> parse -> fetch DS -> process new/updated files -> Tika -> embed -> update DB
    """
    start_time = time.time()
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(0)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"GCP ingest - parsed event_data: {event_data}")
    except Exception as e:
        log.error("Failed to parse DATA_SOURCE_CONFIG for GCP.")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    # 1) Mark DataSource as 'processing'
    try:
        res = psql.update_data_source_by_id(data_source_id, status="processing")
        log.info(f"DB update response: {res}")
        log.info(f"DataSource {data_source_id} status updated to 'processing'.")
    except Exception as e:
        log.error(f"Failed to update DataSource {data_source_id} to 'processing': {e}")
        res = psql.update_data_source_by_id(data_source_id, status="failed")
        log.info(f"DB update response: {res}")
        sys.exit(1)

    try:
        # 2) Retrieve the DataSource from DB
        ds = psql.get_data_source_details(data_source_id)
        log.info(f"Retrieved DataSource from DB: {ds}")
        if not ds:
            log.error(f"No DataSource found for id={data_source_id}. Exiting.")
            return ("No DataSource found for id={}".format(data_source_id), 404)
        if ds["sourceType"] != "gcpBucket":
            log.error(f"DataSource {data_source_id} is {ds['sourceType']}, not gcpBucket. Exiting.")
            res = psql.update_data_source_by_id(data_source_id, status="failed")
            log.info(f"DB update response: {res}")
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not gcpBucket.", 400)

        # 3) Validate credentials and bucket
        credential_json = ds.get("gcpCredentialJson")
        bucket_name = ds.get("gcpBucketName")
        folder_prefix = ds.get("gcpFolderPath") or ""
        if not credential_json or not bucket_name:
            log.error("Missing GCP credentials or bucket in DataSource record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Missing gcpCredentialJson or gcpBucketName", 400)

        # 4) Parse last sync time
        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
                log.info(f"Parsed last_sync_dt: {last_sync_dt}")
            except Exception as e:
                log.warning(f"Failed to parse lastSyncTime: {last_sync_time}. Proceeding without it.")
                res = psql.update_data_source_by_id(data_source_id, status="failed")
                log.info(f"DB update response: {res}")
                last_sync_dt = None

        # 5) Connect to GCP Storage
        log.info("Connecting to GCP storage")
        client = connect_gcp_storage(credential_json)
        bucket = client.bucket(bucket_name)

        # 6) List all blobs
        all_blobs = list(bucket.list_blobs(prefix=folder_prefix))
        log.info(f"Found {len(all_blobs)} blobs in gs://{bucket_name}/{folder_prefix}")
        if not all_blobs:
            log.warning(f"No blobs found in gs://{bucket_name}/{folder_prefix}")
        else:
            log.info("Listing all blobs to ensure recursive fetching.")
        
        for blob in all_blobs:
            log.debug(f"Blob found: {blob.name}, Last Modified: {blob.updated}")

        # 7) Gather existing files from DB
        existing_files = psql.fetch_files_by_data_source_id(data_source_id)
        db_file_keys = set(ef["id"] for ef in existing_files)
        log.info(f"Retrieved {len(existing_files)} existing files from DB.")

        # 8) Prepare for new/updated logic
        sub_id = None
        if project_id:
            proj_details = psql.get_project_details(project_id)
            if proj_details:
                sub_id = proj_details.get("subscriptionId")
        sub_for_hash = sub_id if sub_id else "no_subscription"

        new_files = []
        updated_files = []
        gcp_file_keys = set()

        # Identify new vs updated files
        for blob in all_blobs:
            if blob.name.endswith("/"):
                # skip "directory placeholders"
                log.debug(f"Skipping directory placeholder: {blob.name}")
                continue
            gcp_key = blob.name
            last_modified = blob.updated or datetime.now(CENTRAL_TZ)
            if not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            gcp_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((blob, gcp_key, last_modified))
                log.debug(f"Identified new file: {gcp_key} -> file_key: {file_key}")
            else:
                # Compare last_modified with last_sync_dt
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))
                    log.debug(f"Identified updated file: {gcp_key} -> file_key: {file_key}")
                elif not last_sync_dt:
                    updated_files.append((blob, gcp_key, last_modified))
                    log.debug(f"No last_sync_time, so marking updated file: {gcp_key} -> file_key: {file_key}")

        # Identify removed files (files in DB but no longer in storage)
        removed_keys = db_file_keys - gcp_file_keys
        log.info(f"Identified {len(removed_keys)} removed files.")
        for r_key in removed_keys:
            log.debug(f"Removing file_key: {r_key} from DB and Pinecone.")
            remove_file_from_db_and_pinecone(r_key, data_source_id, project_id, namespace=project_id)

        log.info(f"New files to process: {len(new_files)}")
        log.info(f"Updated files to process: {len(updated_files)}")
        to_process = new_files + updated_files
        log.info(f"Total files to process: {len(to_process)}")

        # 9) If no new or updated files -> mark DS as processed
        if not to_process:
            now_dt = datetime.now(CENTRAL_TZ)
            psql.update_data_source_by_id(
                data_source_id,
                status="processed",
                lastSyncTime=now_dt.isoformat(),
                nextSyncTime=compute_next_sync_time(now_dt, ds.get("syncOption")).isoformat(),
            )
            log.info("No new or updated files to process. DataSource status set to 'processed'.")
            return ("No new/updated files, done", 200)

        # 10) Tika server check
        if not SERVER_URL:
            log.error("SERVER_URL is not set. Cannot proceed with Tika processing.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("No SERVER_URL", 500)
        
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
        # Prepare Tika request headers
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)
        total_processed_files = 0

        # 11) Process each file
        for (blob, gcp_key, last_modified) in to_process:
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, gcp_key)
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(gcp_key)

            log.info(f"Processing file: {gcp_key} with file_key: {file_key}")

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
                    source_type="gcpBucket",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
                log.debug(f"Added new file record in DB with key={file_key}")
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())
                log.debug(f"Updated existing file record in DB to 'processing': {file_key}")

            # 12) Determine file extension and local path
            extension = "pdf"
            if "." in gcp_key:
                extension = gcp_key.rsplit(".", 1)[-1].lower()
            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")

            # 13) Download from GCP
            try:
                blob.download_to_filename(local_tmp_path)
                log.info(f"Successfully downloaded blob: gs://{bucket_name}/{gcp_key} to {local_tmp_path}")
            except Exception as e:
                log.error(f"Failed to download gs://{bucket_name}/{gcp_key}: {e}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            def process_local_file(temp_path, file_extension):
                """
                Local helper to handle PDF splitting, or docx/xlsx/pptx -> PDF conversion + splitting.
                """
                if file_extension == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    log.debug(f"Splitting PDF file: {temp_path}")
                    return file_processor.split_pdf(pdf_data)
                elif file_extension in ["docx", "xlsx", "pptx"]:
                    log.debug(f"Converting {file_extension} file to PDF: {temp_path}")
                    pdf_data = convert_to_pdf(temp_path, file_extension)
                    if pdf_data:
                        log.debug(f"Splitting converted PDF for file: {temp_path}")
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError(f"Conversion to PDF failed for file {temp_path}")
                else:
                    raise ValueError(f"Unsupported file format in GCP ingestion logic: {file_extension}")

            # 14) Convert & split
            try:
                final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
                log.info(f"Processed file {base_name}: extracted {final_num_pages} pages.")
            except Exception as e:
                log.error(f"Failed processing {base_name} as {extension}: {str(e)}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                if os.path.exists(local_tmp_path):
                    os.remove(local_tmp_path)
                    log.debug(f"Removed temporary file: {local_tmp_path}")
                continue

            # Remove temp file
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
                log.debug(f"Removed temporary file: {local_tmp_path}")

            # 15) Tika + embeddings
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
                    )
                )
                log.info(f"Completed Tika processing for file {base_name}.")
            except Exception as e:
                log.error(f"Failed Tika/embedding for {base_name}: {e}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            # 16) Mark file as processed in DB
            psql.update_file_by_id(
                file_key,
                status="processed",
                pageCount=len(results),
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )
            log.info(f"Updated file {base_name} to 'processed' with {len(results)} pages embedded.")

            # 17) Publish credit usage
            used_credits = len(results) * 1.5
            message = json.dumps({
                "subscription_id": sub_id,
                "data_source_id": data_source_id,
                "project_id": project_id,
                "creditsUsed": used_credits,
            })
            try:
                google_pub_sub.publish_messages_with_retry_settings(
                    GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                )
                log.info(f"Published credit usage for file {base_name}: {used_credits} credits.")
            except Exception as e:
                log.warning(f"Publish usage failed for file {base_name}: {str(e)}")

            total_processed_files += 1

        # 18) Mark DataSource as processed
        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=compute_next_sync_time(final_now_dt, ds.get("syncOption")).isoformat(),
        )
        log.info(f"Ingestion job completed. Processed {total_processed_files} files.")

        elapsed_time = time.time() - start_time
        log.info(f"Total ingestion time: {elapsed_time:.2f} seconds.")
        return ("OK", 200)

    except Exception as e:
        log.error(f"Error in GCP ingestion flow: {e}")
        psql.update_data_source_by_id(data_source_id, status="error")
        return (str(e), 500)

def shutdown_handler(sig, frame):
    """Handle SIGINT or SIGTERM to gracefully exit."""
    log.info(f"Caught signal {signal.strsignal(sig)}. Exiting.")
    sys.exit(0)

# -----------------------------------------------------------------------------------
# MAIN EXECUTION GUARD
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Ensure UPLOADS_FOLDER is created
    if not os.path.exists(UPLOADS_FOLDER):
        try:
            os.makedirs(UPLOADS_FOLDER, exist_ok=True)
            log.info(f"Created UPLOADS_FOLDER at {UPLOADS_FOLDER}")
        except Exception as e:
            log.error(f"Failed to create UPLOADS_FOLDER at {UPLOADS_FOLDER}: {e}")
            sys.exit(1)

    log.debug(f"UPLOADS_FOLDER is set to: {UPLOADS_FOLDER}")

    if not SERVER_URL:
        log.error("SERVER_URL is not set correctly.")
        sys.exit(1)
    log.debug(f"SERVER_URL is set to: {SERVER_URL}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Run the ingestion job
    response, status_code = run_job()
    log.info(f"Ingestion job response: {response} with status code: {status_code}")

    elapsed_time = time.time() - start_time
    log.info(f"Total script execution time: {elapsed_time:.2f} seconds.")

else:
    # If the script is imported, we still handle SIGTERM
    signal.signal(signal.SIGTERM, shutdown_handler)
