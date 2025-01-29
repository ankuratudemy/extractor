#!/usr/bin/env python3
# s3_ingest.py

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
from shared import psql, google_pub_sub, google_auth, file_processor  # or rename as needed
from pinecone.grpc import PineconeGRPC as Pinecone

# If using Vertex AI for embeddings (or replace w/ your approach)
from vertexai.preview.language_models import TextEmbeddingModel

# Additional import for S3
import boto3
from botocore.config import Config

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

SERVER_DOMAIN = os.environ.get("SERVER_URL")  # e.g. "mydomain.com"
SERVER_URL = f"https://{SERVER_DOMAIN}/tika" if SERVER_DOMAIN else None

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

CENTRAL_TZ = ZoneInfo("America/Chicago")

# ----------------------------------------------------------------------------
# INITIALIZE PINECONE
# ----------------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

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
    Generates an MD5 hash from the JSON-dumped data,
    ensuring uniqueness for user-specific combos.
    """
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    md5_hash = hashlib.md5(combined_string.encode("utf-8")).hexdigest()
    return md5_hash


def compute_next_sync_time(from_date: datetime, sync_option: str = None) -> datetime:
    """
    Computes the next sync time based on sync_option: 'hourly'|'daily'|'weekly'|'monthly'.
    Fallback to 1 hour if no valid syncOption is provided.
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
        # default
        return from_date + timedelta(hours=1)


def remove_file_from_db_and_pinecone(file_id, ds_id, project_id, namespace):
    """
    Removes references in DB + Pinecone for the given file_id => consistent with your approach.
    """
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        for ids in index.list(prefix=vector_id_prefix, namespace=namespace):
            log.info(f"S3 Pinecone Ids to delete: {ids}")
            index.delete(ids=ids, namespace=namespace)
        log.info(f"Removed Pinecone vectors with prefix={vector_id_prefix}")
    except Exception as e:
        log.exception(f"Error removing vector {vector_id_prefix} from Pinecone:")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception as e:
        log.exception(f"Error removing DB File {file_id}:")


def convert_to_pdf(file_path, file_extension):
    """
    Uses LibreOffice to convert to PDF if needed.
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
    if not SERVER_URL:
        raise ValueError("SERVER_URL is not set, cannot call Tika service.")

    url = SERVER_URL
    log.info(f"Starting async processing of {len(pages)} pages w/ Tika at {url}.")
    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_put_page(session, url, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)

    log.info("All pages extracted. Now chunk + embed + upsert to Pinecone.")
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

        overlap_str = chunk_str[-overlap_chars:] if len(chunk_str) > overlap_chars else chunk_str
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

            # If adding this chunk exceeds limits, flush the batch
            if ((batch_text_count + 1) > max_batch_texts) or (
                batch_token_count + chunk_token_len > max_batch_tokens
            ):
                if batch:
                    await process_embedding_batch(
                        batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
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

    # We'll compute the sparse vector **per chunk**:
    # That means we handle each (chunk, embedding) pair individually, rather than
    # combining them into a single "texts" argument (which caused the error).
    vectors = []

    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num,
            "sourceType": "s3",
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            ),
        }
        vectors.append({
            "id": doc_id,
            "values": embedding,
            "sparse_values": sparse_values,
            "metadata": metadata
        })

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)


async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    """
    Using Vertex AI for embeddings (as in your existing code).
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


# ----------------------------------------------------------------------------
# S3 LISTING HELPERS
# ----------------------------------------------------------------------------

def get_s3_client(access_key, secret_key, region):
    """
    Returns a Boto3 S3 client with the provided credentials and region.
    """
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    # Optionally configure timeouts/retries
    config = Config(
        retries = {
            'max_attempts': 10,
            'mode': 'standard'
        }
    )
    return session.client('s3',region_name=region, config=config)


def list_s3_objects(s3_client, bucket_name, prefix=""):
    """
    Lists all objects in the given bucket under `prefix` (if provided).
    Returns a list of dict: { 'Key': ..., 'LastModified': ..., ... }
    """
    all_items = []
    continuation_token = None

    while True:
        kwargs = {
            'Bucket': bucket_name,
            'Prefix': prefix
        }
        if continuation_token:
            kwargs['ContinuationToken'] = continuation_token

        resp = s3_client.list_objects_v2(**kwargs)
        contents = resp.get('Contents', [])
        for obj in contents:
            all_items.append(obj)

        if resp.get('IsTruncated'):
            continuation_token = resp.get('NextContinuationToken')
        else:
            break

    return all_items


def download_s3_object(s3_client, bucket_name, key, local_path):
    """
    Downloads the object from S3 to `local_path`.
    """
    s3_client.download_file(bucket_name, key, local_path)


# ----------------------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------------------

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
        log.info(f"data source from db: {ds}")
        if not ds:
            log.info(f"No DS found with id={data_source_id}")
            return ("No DataSource found", 404)
        if ds["sourceType"] != "s3":
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not s3.", 400)

        # We do not refresh tokens for S3. We'll read credentials from ds:
        access_key = ds.get("s3AccessKey")
        secret_key = ds.get("s3SecretKey")
        region = ds.get("s3BucketRegion")
        bucket_name = ds.get("s3BucketName")
        folder_prefix = ds.get("s3FolderPath") or ""

        if not (access_key and secret_key and bucket_name):
            log.error("Missing S3 credentials or bucket in ds record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Missing S3 credentials/bucket", 400)

        # lastSyncTime
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

        # Connect to S3
        s3_client = get_s3_client(access_key, secret_key, region)

        # 1) List objects from S3
        s3_items = list_s3_objects(s3_client, bucket_name, folder_prefix)
        log.info(f"Found {len(s3_items)} total objects in s3://{bucket_name}/{folder_prefix}")

        # 2) DB existing files
        existing_files = psql.fetch_files_by_data_source_id(data_source_id)
        db_file_keys = set(ef["id"] for ef in existing_files)

        sub_id = None
        if project_id:
            proj_details = psql.get_project_details(project_id)
            if proj_details:
                sub_id = proj_details.get("subscriptionId")
        sub_for_hash = sub_id if sub_id else "no_subscription"

        s3_key_to_md5 = {}
        s3_file_keys = set()

        # Gather new/updated
        new_files = []
        updated_files = []

        for obj in s3_items:
            s3key = obj["Key"]
            last_modified = obj["LastModified"]  # datetime
            if not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            # Build unique file_key
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, s3key)
            s3_key_to_md5[s3key] = file_key
            s3_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((s3key, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((s3key, last_modified))
                elif not last_sync_dt:
                    # If no last sync, treat as updated to re-embed
                    updated_files.append((s3key, last_modified))

        removed_keys = db_file_keys - s3_file_keys
        log.info(f"Removed keys from DB: {removed_keys}")
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
            return ("No new/updated files, done.", 200)

        # 3) Tika / embedding
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

        db_file_keys_list = set(db_file_keys)  # mutable set to track inserts

        for (s3key, obj_last_modified) in to_process:
            file_key = s3_key_to_md5[s3key]
            now_dt = datetime.now(CENTRAL_TZ)

            # If not in DB, add
            if file_key not in db_file_keys_list:
                base_name = os.path.basename(s3key)
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
                    source_type="s3",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            # Download object
            extension = "pdf"
            if "." in s3key:
                extension = s3key.rsplit(".", 1)[-1].lower()
            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")

            try:
                download_s3_object(s3_client, bucket_name, s3key, local_tmp_path)
            except Exception as e:
                log.exception(f"Failed to download s3://{bucket_name}/{s3key}:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            # Process locally
            def process_local_file(temp_path, file_extension):
                if file_extension == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_extension in ["docx", "xlsx", "pptx", "odt", "etc"]:
                    pdf_data = convert_to_pdf(temp_path, file_extension)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed for S3 file.")
                raise ValueError("Unsupported file format in S3 ingestion logic")

            base_name = os.path.basename(s3key)
            try:
                final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
            except Exception as e:
                log.error(f"Failed processing {base_name} as {extension}: {str(e)}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
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
                        last_modified=obj_last_modified,
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding step for {base_name}:")
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
                log.warning(f"Publish usage failed for S3 file {base_name}: {str(e)}")

            del final_pages, results

        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=compute_next_sync_time(final_now_dt, ds.get("syncOption")).isoformat(),
        )
        return ("OK", 200)

    except Exception as e:
        log.exception("Error in S3 ingestion flow:")
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
