# BM25 + sparse vector utilities (with no spaCy usage)
import os
import io
import ssl
import sys
import signal
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import asyncio
import aiohttp
import tiktoken
import subprocess
from dateutil import parser
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
from typing import List, Tuple
from shared.bm25 import (
    tokenize_document,
    publish_partial_bm25_update,
    compute_bm25_sparse_vector,
    get_project_vocab_stats,
)
from pinecone.grpc import PineconeGRPC as Pinecone
from vertexai.preview.language_models import TextEmbeddingModel

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

# For XLSX processing via separate Flask API:
XLSX_SERVER_URL = os.environ.get("XLSX_SERVER_URL", None)
XLSX_SERVER_ENDPOINT = f"https://{XLSX_SERVER_URL}/process_xlsx"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int) -> dict:
    combined_text = " ".join(file_texts)
    tokens = tokenize_document(combined_text)
    publish_partial_bm25_update(project_id, tokens, is_new_doc=True)
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
            log.warning(
                f"Error PUT page {page_num}: {str(e)}, retry {retries+1}/{max_retries}"
            )
            retries += 1
            await asyncio.sleep(1)
    raise RuntimeError(f"Failed after {max_retries} tries for page {page_num}")

async def process_pages_async(
    pages, headers, filename, namespace, file_id, data_source_id, last_modified, sourceType
):
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
        results, filename, namespace, file_id, data_source_id, last_modified, sourceType
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

async def process_embedding_batch(
    batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values, sourceType
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
            "sourceType": sourceType,
            "dataSourceId": data_source_id,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified)
            ),
        }
        vectors.append(
            {
                "id": doc_id,
                "values": embedding,
                "sparse_values": sparse_values,
                "metadata": metadata,
            }
        )

    log.info(f"Upserting {len(vectors)} vectors to Pinecone for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)

async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


async def create_and_upload_embeddings_in_batches(
    results, filename, namespace, file_id, data_source_id, last_modified, sourceType
):
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    # Single-file BM25 update for all pages
    all_file_texts = [text for (text, _page_num) in results if text.strip()]
    sparse_values = get_sparse_vector(all_file_texts, namespace, 300)

    for text, page_num in results:
        if not text.strip():
            continue
        text_chunks = chunk_text(text, max_tokens=2048, overlap_chars=2000)
        for i, chunk in enumerate(text_chunks):
            chunk_token_len = len(enc.encode(chunk))
            # If adding this chunk exceeds any limit, process the current batch
            if ((batch_text_count + 1) > max_batch_texts) or (batch_token_count + chunk_token_len > max_batch_tokens):
                if batch:
                    await process_embedding_batch(
                        batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values, sourceType
                    )
                    batch.clear()
                    batch_text_count = 0
                    batch_token_count = 0

            if ((batch_text_count + 1) <= max_batch_texts) and ((batch_token_count + chunk_token_len) <= max_batch_tokens):
                batch.append((chunk, page_num, i))
                batch_text_count += 1
                batch_token_count += chunk_token_len
            else:
                log.warning("Chunk too large or logic error preventing batch addition.")
                continue

    if batch:
        await process_embedding_batch(
            batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values, sourceType
        )
        batch.clear()


# ----------------------------------------------------------------------------
# NEW ASYNC XLSX HELPER => We'll do parallel calls to XLSX_SERVER_ENDPOINT
# ----------------------------------------------------------------------------
async def _call_xlsx_endpoint(session, xlsx_flask_url, file_path, base_name, project_id, data_source_id, sub_for_hash):
    """
    A helper that:
    1) Reads the XLSX from local disk
    2) Posts it to XLSX_SERVER_ENDPOINT
    3) Returns (status_code, response_text or response_json)
    """
    # We'll set a large timeout if needed; or you can use custom handling.
    timeout = aiohttp.ClientTimeout(total=3600)  # up to 1h for large XLSX

    with open(file_path, "rb") as f:
        file_content = f.read()

    data = {
        "project_id": project_id,
        "data_source_id": data_source_id,
        "sub_id": sub_for_hash,
    }

    form_data = aiohttp.FormData()
    form_data.add_field("file", file_content, filename=base_name, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    for k, v in data.items():
        form_data.add_field(k, v)
    try:
        xlsx_bearer_token = google_auth.impersonated_id_token(serverurl=XLSX_SERVER_URL).json()["token"]
    except Exception as e:
        log.exception("Failed to obtain impersonated ID token:")
        psql.update_data_source_by_id(
            data_source_id, status="error", updatedAt=datetime.now(CENTRAL_TZ).isoformat()
        )
        return ("Failed to obtain impersonated ID token.", 500)

    headers = {
        "Authorization": f"Bearer {xlsx_bearer_token}",
        "X-Tika-PDFOcrStrategy": "auto",
        "Accept": "text/plain",
    }
    async with session.post(xlsx_flask_url, data=form_data,headers=headers, timeout=timeout) as resp:
        status = resp.status
        try:
            js = await resp.json()
        except:
            js = await resp.text()
        return (status, js)

async def process_xlsx_blob(
    session,  # shared ClientSession
    file_key,
    base_name,
    local_tmp_path,
    project_id,
    data_source_id,
    sub_for_hash,
    sub_id
):
    """
    This function processes a single XLSX file in parallel:
      1) Call the XLSX Flask endpoint
      2) Return usage info or errors
    """
    # We'll default to "failed" unless we get a 200
    final_status = "failed"
    usage_credits = 0
    error_msg = ""

    # Attempt the POST
    xlsx_flask_url = XLSX_SERVER_ENDPOINT  # global var
    try:
        # Call the XLSX endpoint
        status_code, response_data = await _call_xlsx_endpoint(
            session,
            xlsx_flask_url,
            local_tmp_path,
            base_name,
            project_id,
            data_source_id,
            sub_for_hash
        )

        if os.path.exists(local_tmp_path):
            os.remove(local_tmp_path)

        if status_code == 200:
            final_status = "processed"
            # parse usage if we want
            if isinstance(response_data, dict):
                usage_credits = response_data.get("chunks_processed", 0) * 1.5
            else:
                # if the response is text instead of JSON
                usage_credits = 0
        else:
            # keep final_status = "failed"
            error_msg = f"XLSX ingest error: status={status_code}, resp={response_data}"

    except Exception as e:
        if os.path.exists(local_tmp_path):
            os.remove(local_tmp_path)
        error_msg = f"Exception calling XLSX ingestion: {str(e)}"

    return final_status, usage_credits, error_msg


def shutdown_handler(sig, frame):
    log.info(f"Caught signal {signal.strsignal(sig)}. Exiting.")
    sys.exit(0)