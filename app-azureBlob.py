#!/usr/bin/env python3
# azure_ingest.py

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

# If using Vertex AI for embeddings
from vertexai.preview.language_models import TextEmbeddingModel

# Azure blob
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

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

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

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
    return hashlib.md5(combined_string.encode("utf-8")).hexdigest()

def compute_next_sync_time(from_date: datetime, sync_option: str = None) -> datetime:
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
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        for ids in index.list(prefix=vector_id_prefix, namespace=namespace):
            log.info(f"Azure Pinecone Ids to delete: {ids}")
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

async def process_pages_async(pages, headers, filename, namespace, file_id, data_source_id, last_modified):
    if not SERVER_URL:
        raise ValueError("SERVER_URL is not set, cannot call Tika.")
    url = SERVER_URL
    log.info(f"Starting async Tika processing of {len(pages)} pages.")
    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_put_page(session, url, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)

    log.info("All pages extracted. Now chunk, embed, and upsert to Pinecone.")
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
            chunk_token_len = len(enc.encode(chunk))
            if ((batch_text_count + 1) > max_batch_texts) or (
                batch_token_count + chunk_token_len > max_batch_tokens
            ):
                if batch:
                    await process_embedding_batch(
                        batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
                    )
                    batch.clear()
                    batch_text_count = 0
                    batch_token_count = 0

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

async def process_embedding_batch(batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values):
    texts = [item[0] for item in batch]
    embeddings = await get_google_embedding(texts)
    vectors = []
    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num,
            "sourceType": "azureBlob",
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            ),
        }
        vectors.append({"id": doc_id, "values": embedding,"sparse_values": sparse_values, "metadata": metadata})

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)

async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
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
        log.info(f"Azure ingest - parsed event_data: {event_data}")
    except Exception as e:
        log.exception("Failed to parse DATA_SOURCE_CONFIG for Azure.")
        sys.exit(1)

    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    psql.update_data_source_by_id(data_source_id, status="processing")

    try:
        ds = psql.get_data_source_details(data_source_id)
        log.info(f"DataSource from db: {ds}")
        if not ds:
            return ("No DataSource found for id={}".format(data_source_id), 404)
        if ds["sourceType"] != "azureBlob":
            return (f"DataSource {data_source_id} is {ds['sourceType']}, not azureBlob.", 400)

        azure_account_name = ds.get("azureStorageAccountName")
        azure_account_key = ds.get("azureStorageAccountKey")
        container_name = ds.get("azureContainerName")
        folder_prefix = ds.get("azureFolderPath") or ""

        if not azure_account_name or not azure_account_key or not container_name:
            log.error("Missing Azure credentials or container in ds record.")
            psql.update_data_source_by_id(data_source_id, status="error")
            return ("Missing azure credentials or containerName", 400)

        last_sync_time = ds.get("lastSyncTime") or event_data.get("lastSyncTime")
        last_sync_dt = None
        if isinstance(last_sync_time, datetime):
            last_sync_dt = ensure_timezone_aware(last_sync_time)
        elif isinstance(last_sync_time, str):
            try:
                last_sync_dt = ensure_timezone_aware(parser.isoparse(last_sync_time))
            except:
                last_sync_dt = None

        # Connect to Azure Blob
        conn_str = f"DefaultEndpointsProtocol=https;AccountName={azure_account_name};AccountKey={azure_account_key};EndpointSuffix=core.windows.net"
        service_client = BlobServiceClient.from_connection_string(conn_str)
        container_client = service_client.get_container_client(container_name)

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
        azure_file_keys = set()

        # list blobs with prefix
        blob_list = container_client.list_blobs(name_starts_with=folder_prefix)
        all_blobs = list(blob_list)

        log.info(f"Found {len(all_blobs)} objects in container={container_name}, prefix={folder_prefix}")

        for blob in all_blobs:
            if blob.name.endswith("/"):
                continue
            blob_name = blob.name
            last_modified = blob.last_modified
            if not last_modified.tzinfo:
                last_modified = last_modified.replace(tzinfo=CENTRAL_TZ)

            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, blob_name)
            azure_file_keys.add(file_key)

            if file_key not in db_file_keys:
                new_files.append((blob, blob_name, last_modified))
            else:
                if last_sync_dt and last_modified > last_sync_dt:
                    updated_files.append((blob, blob_name, last_modified))
                elif not last_sync_dt:
                    updated_files.append((blob, blob_name, last_modified))

        removed_keys = db_file_keys - azure_file_keys
        log.info(f"Removed keys from DB: {removed_keys}")
        for r_key in removed_keys:
            remove_file_from_db_and_pinecone(r_key, data_source_id, project_id, namespace=project_id)

        log.info(f"New files => {len(new_files)}, Updated => {len(updated_files)}")
        to_process = new_files + updated_files
        if not to_process:
            now_dt = datetime.now(CENTRAL_TZ)
            psql.update_data_source_by_id(
                data_source_id,
                status="processed",
                lastSyncTime=now_dt.isoformat(),
                nextSyncTime=compute_next_sync_time(now_dt, ds.get("syncOption")).isoformat(),
            )
            return ("No new/updated files, done", 200)

        if not SERVER_URL:
            log.error("SERVER_URL is not set. Can't proceed with Tika.")
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

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Tika-PDFOcrStrategy": "auto",
            "Accept": "text/plain",
        }

        loop = get_event_loop()
        db_file_keys_list = set(db_file_keys)

        for (blob, blob_name, last_modified) in to_process:
            file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, blob_name)
            now_dt = datetime.now(CENTRAL_TZ)
            base_name = os.path.basename(blob_name)

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
                    source_type="azureBlob",
                    data_source_id=data_source_id,
                    project_id=project_id,
                )
                db_file_keys_list.add(file_key)
            else:
                psql.update_file_by_id(file_key, status="processing", updatedAt=now_dt.isoformat())

            extension = "pdf"
            if "." in blob_name:
                extension = blob_name.rsplit(".", 1)[-1].lower()

            local_tmp_path = os.path.join(UPLOADS_FOLDER, f"{file_key}.{extension}")

            try:
                with open(local_tmp_path, "wb") as f:
                    download_stream = container_client.download_blob(blob.name)
                    f.write(download_stream.readall())
            except Exception as e:
                log.exception(f"Failed to download azure blob: {blob_name}")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            def process_local_file(temp_path, file_extension):
                if file_extension == "pdf":
                    with open(temp_path, "rb") as f_in:
                        pdf_data = f_in.read()
                    return file_processor.split_pdf(pdf_data)
                elif file_extension in ["docx", "xlsx", "pptx"]:
                    pdf_data = convert_to_pdf(temp_path, file_extension)
                    if pdf_data:
                        return file_processor.split_pdf(pdf_data)
                    else:
                        raise ValueError("Conversion to PDF failed for Azure file.")
                raise ValueError("Unsupported file format in Azure ingestion logic")

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
                        last_modified=last_modified,
                    )
                )
            except Exception as e:
                log.exception(f"Failed Tika/embedding for {base_name}:")
                psql.update_file_by_id(file_key, status="failed", updatedAt=now_dt.isoformat())
                continue

            psql.update_file_by_id(
                file_key,
                status="processed",
                pageCount=len(results),
                updatedAt=datetime.now(CENTRAL_TZ).isoformat(),
            )

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
            except Exception as e:
                log.warning(f"Publish usage failed for Azure file {base_name}: {str(e)}")

        final_now_dt = datetime.now(CENTRAL_TZ)
        psql.update_data_source_by_id(
            data_source_id,
            status="processed",
            lastSyncTime=final_now_dt.isoformat(),
            nextSyncTime=compute_next_sync_time(final_now_dt, ds.get("syncOption")).isoformat(),
        )
        return ("OK", 200)

    except Exception as e:
        log.exception("Error in Azure ingestion flow:")
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
        log.error("SERVER_URL environment variable is not set.")
        sys.exit(1)
    log.debug(f"SERVER_URL is set to: {SERVER_URL}")

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    run_job()
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
