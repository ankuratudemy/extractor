import os
import io
import ssl
import sys
import signal
import json
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import asyncio
import aiohttp
import tiktoken
import subprocess
from dateutil import parser
from typing import List, Tuple
from shared.logging_config import log
from shared import psql, google_pub_sub, google_auth, file_processor
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
METADATA_SERVER_URL = os.environ.get("METADATA_SERVER_URL", None)
METADATA_ENDPOINT = f"https://{METADATA_SERVER_URL}/get_metadata" if METADATA_SERVER_URL else None

CENTRAL_TZ = ZoneInfo("America/Chicago")
SERVER_DOMAIN = os.environ.get("SERVER_URL")
SERVER_URL = f"https://{SERVER_DOMAIN}/tika" if SERVER_DOMAIN else None

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

# For XLSX processing via separate Flask API:
XLSX_SERVER_URL = os.environ.get("XLSX_SERVER_URL", None)
XLSX_SERVER_ENDPOINT = f"https://{XLSX_SERVER_URL}/process_spreadsheet"
log.info(f"METADATA SERVER ENDPOINT {METADATA_ENDPOINT}")
log.info(f"XLSX SERVER ENDPOINT {XLSX_SERVER_ENDPOINT}")
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
METADATA_SESSION = None

async def fetch_additional_metadata(file_path: str, file_id: str, project_id: str, sub_id: str) -> dict:
    global METADATA_SESSION

    # If we don't have a metadata server configured, just skip
    if not METADATA_ENDPOINT:
        log.info("[Metadata] METADATA_SERVER_URL not set. Skipping metadata fetch.")
        return {}

    # If the global session is None or closed, create/recreate it with high timeouts
    if METADATA_SESSION is None or METADATA_SESSION.closed:
        # e.g. 10 minutes total, 30s connect, 30s DNS resolution, 600s read
        timeout = aiohttp.ClientTimeout(total=600, connect=30, sock_connect=30, sock_read=600)
        METADATA_SESSION = aiohttp.ClientSession(timeout=timeout)

    # Load file into memory
    if not os.path.exists(file_path):
        log.warning(f"[Metadata] File {file_path} does not exist. Cannot fetch metadata.")
        return {}

    with open(file_path, "rb") as f:
        file_content = f.read()

    form_data = aiohttp.FormData()
    form_data.add_field("file", file_content, filename=os.path.basename(file_path))
    form_data.add_field("project_id", str(project_id))
    form_data.add_field("sub_id", str(sub_id))

    # If you need a Bearer token for METADATA_ENDPOINT:
    try:
        metadata_bearer_token = google_auth.impersonated_id_token(serverurl=METADATA_SERVER_URL).json()["token"]
    except Exception as exc:
        log.exception("[Metadata] Failed to get impersonated token for metadata server:")
        return {}

    headers = {
        "Authorization": f"Bearer {metadata_bearer_token}",
        "Accept": "application/json"
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with METADATA_SESSION.post(METADATA_ENDPOINT, data=form_data, headers=headers) as resp:
                if resp.status == 429 or (500 <= resp.status < 600):
                    log.warning(
                        f"[Metadata] Received status={resp.status} from /metadata. "
                        f"Retrying {attempt+1}/{max_retries}."
                    )
                    await asyncio.sleep(2 * (attempt + 1))
                    continue

                if resp.status >= 400:
                    log.error(f"[Metadata] Non-success HTTP status: {resp.status}")
                    return {}

                # Attempt to parse JSON
                response_data = await resp.json()
                if not response_data:
                    log.info("[Metadata] /metadata returned empty JSON.")
                    return {}
                else:
                    log.info(f"[Metadata] Successfully fetched metadata for file_id={file_id}.")
                    return response_data

        except Exception:
            log.exception("[Metadata] Exception in fetch_additional_metadata:")
            await asyncio.sleep(2 * (attempt + 1))

    # If we exhausted retries
    log.error("[Metadata] Max retries exceeded for /metadata call.")
    return {}
def parse_last_sync_time(last_sync_time_value):
    """
    Safely parse a lastSyncTime value (string or datetime) into a timezone-aware datetime or return None.
    """
    if isinstance(last_sync_time_value, datetime):
        return ensure_timezone_aware(last_sync_time_value)
    if isinstance(last_sync_time_value, str):
        try:
            return ensure_timezone_aware(parser.isoparse(last_sync_time_value))
        except:
            return None
    return None

def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int) -> dict:
    """
    Convert the combined text of a document into a BM25 sparse vector for Pinecone,
    which helps with hybrid search (dense + sparse).
    """
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
    """
    Remove the file from the PostgreSQL database and also remove any vectors
    in Pinecone that have the prefix ds_id#file_id#.
    """
    vector_id_prefix = f"{ds_id}#{file_id}#"
    try:
        for ids in index.list(prefix=vector_id_prefix, namespace=namespace):
            log.info(f"Pinecone IDs to delete: {ids}")
            index.delete(ids=ids, namespace=namespace)
        log.info(f"Removed Pinecone vectors with prefix={vector_id_prefix}")
    except Exception:
        log.exception(f"Error removing vector {vector_id_prefix} from Pinecone:")

    try:
        psql.delete_file_by_id(file_id)
        log.info(f"Removed DB File {file_id}")
    except Exception:
        log.exception(f"Error removing DB File {file_id}:")

def remove_missing_files(db_file_keys, remote_file_keys, data_source_id, project_id, namespace=None):
    """
    Remove from DB and Pinecone any files that no longer exist in the remote source.
    """
    removed_keys = db_file_keys - remote_file_keys
    log.info(f"Removed keys: {removed_keys}")
    for r_key in removed_keys:
        remove_file_from_db_and_pinecone(r_key, data_source_id, project_id, namespace=namespace or project_id)


def convert_to_pdf(file_path, file_extension):
    """
    Convert docx/pptx to PDF using LibreOffice, returning the PDF bytes.
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
            return pdf_data
        else:
            log.error("PDF file not found after conversion.")
            return None
    except subprocess.CalledProcessError:
        log.exception("Conversion to PDF failed:")
        return None
    except Exception:
        log.exception("Error during PDF conversion:")
        return None


async def _async_put_page(session, url, page_data, page_num, headers, max_retries=5):
    """
    Upload a single PDF page (as BytesIO) to the Tika server, handle 429/5xx retries,
    and return the extracted text.
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
                if resp.status == 429 or (500 <= resp.status < 600):
                    # Retry on 429 or any 5xx
                    log.warning(
                        f"[Tika] Received status={resp.status} for page={page_num}. "
                        f"Retry {retries+1}/{max_retries}..."
                    )
                    retries += 1
                    await asyncio.sleep(2 * retries)
                    continue
                # If 2xx or 4xx other than 429, proceed
                content = await resp.read()
                text_content = content.decode("utf-8", errors="ignore")
                log.info(f"[Tika] Page {page_num} => {len(text_content)} chars extracted.")
                return text_content, page_num

        except (aiohttp.ClientError, ssl.SSLError, asyncio.TimeoutError):
            log.exception(
                f"[Tika] Error PUT page {page_num}, retry {retries+1}/{max_retries}:"
            )
            retries += 1
            await asyncio.sleep(2 * retries)

    # If we get here, we exhausted our retries:
    msg = f"[Tika] Failed to process page {page_num} after {max_retries} retries."
    log.error(msg)
    # Return something indicative
    return "", page_num


async def process_pages_async(
    pages,
    headers,
    filename,
    namespace,
    file_id,
    data_source_id,
    last_modified,
    sourceType,
    additional_metadata=None,  # <--- new
):
    if not SERVER_URL:
        raise ValueError("SERVER_URL is not set, cannot call Tika.")

    async with aiohttp.ClientSession() as session:
        tasks = [
            _async_put_page(session, SERVER_URL, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)

    # results is a list of (extracted_text, page_num)
    await create_and_upload_embeddings_in_batches(
        results,
        filename,
        namespace,
        file_id,
        data_source_id,
        last_modified,
        sourceType,
        additional_metadata=additional_metadata,  # <--- pass here
    )
    return results  # In case we need to pass back to caller



def chunk_text(text, max_tokens=2048, overlap_chars=2000):
    """
    Splits a single string into overlapping chunks based on a max token size.
    Prevents losing semantic context from chunk to chunk.
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

        overlap_str = chunk_str[-overlap_chars:] if len(chunk_str) > overlap_chars else chunk_str
        overlap_tokens = enc.encode(overlap_str)
        overlap_count = len(overlap_tokens)

        start = end - overlap_count
        end = start + max_tokens
    return chunks


async def process_embedding_batch(
    batch,
    filename,
    namespace,
    file_id,
    data_source_id,
    last_modified,
    sparse_values,
    sourceType,
    additional_metadata=None
):
    """
    Sends a batch of chunked text to the embedding model, builds Pinecone
    vectors, then upserts them.
    """
    texts = [item[0] for item in batch]
    embeddings = await get_google_embedding(texts)

    vectors = []
    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        if data_source_id:
            doc_id = f"{data_source_id}#{file_id}#{page_num}#{chunk_idx}"
        else:
            doc_id = f"{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num,
            "sourceType": sourceType,
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat()
                if hasattr(last_modified, "isoformat")
                else str(last_modified)
            ),
            **({"dataSourceId": data_source_id} if data_source_id is not None else {})
        }
        # Attach any additional key-value pairs from the metadata JSON
        if additional_metadata:
            metadata.update(additional_metadata)

        vectors.append(
            {
                "id": doc_id,
                "values": embedding,
                "sparse_values": sparse_values,
                "metadata": metadata,
            }
        )

    log.info(f"[Embed] Upserting {len(vectors)} vectors for file_id={file_id}")
    index.upsert(vectors=vectors, namespace=namespace)


async def get_google_embedding(
    queries, model_name="text-multilingual-embedding-preview-0409"
):
    """
    Uses Vertex AI PaLM to generate embeddings for a list of text queries.
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


async def create_and_upload_embeddings_in_batches(
    results, filename, namespace, file_id, data_source_id, last_modified, sourceType, additional_metadata
):
    """
    Takes raw extracted text from all pages, splits each text into chunks,
    and sends them to process_embedding_batch in manageable batches.
    """
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")
    additional_metadata=None

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
                        sparse_values,
                        sourceType,
                        additional_metadata,
                    )
                    batch.clear()
                    batch_text_count = 0
                    batch_token_count = 0

            # Try to add current chunk
            if ((batch_text_count + 1) <= max_batch_texts) and (
                (batch_token_count + chunk_token_len) <= max_batch_tokens
            ):
                batch.append((chunk, page_num, i))
                batch_text_count += 1
                batch_token_count += chunk_token_len
            else:
                log.warning("[Embed] Chunk too large or logic error prevented batch addition.")
                continue

    # Flush any remaining chunk batch
    if batch:
        await process_embedding_batch(
            batch,
            filename,
            namespace,
            file_id,
            data_source_id,
            last_modified,
            sparse_values,
            sourceType,
        )
        batch.clear()


# ----------------------------------------------------------------------------
# XLSX HELPER => We'll do parallel calls to XLSX_SERVER_ENDPOINT
# ----------------------------------------------------------------------------

async def _call_xlsx_endpoint(
    session,
    xlsx_flask_url,
    file_path,
    base_name,
    project_id,
    data_source_id,
    sub_for_hash,
    max_retries=5,
):
    """
    A helper that:
    1) Reads the XLSX from local disk
    2) Posts it to XLSX_SERVER_ENDPOINT
    3) Retries on 429 or 5xx, up to max_retries
    4) Returns (status_code, response_text or response_json)
    """

    # Read file from disk
    with open(file_path, "rb") as f:
        file_content = f.read()

    data = {
        "project_id": project_id,
        "sub_id": sub_for_hash,
        **({"dataSourceId": data_source_id} if data_source_id is not None else {})
    }

    form_data = aiohttp.FormData()
    form_data.add_field(
        "file",
        file_content,
        filename=base_name,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    for k, v in data.items():
        form_data.add_field(k, v)

    # Obtain a bearer token (outside the retry loop here, unless you need to refresh it each time)
    try:
        xlsx_bearer_token = google_auth.impersonated_id_token(serverurl=XLSX_SERVER_URL).json()["token"]
    except Exception:
        log.exception("[XLSX] Failed to obtain impersonated ID token:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (500, "Failed to obtain impersonated ID token.")

    headers = {
        "Authorization": f"Bearer {xlsx_bearer_token}",
        "X-Tika-PDFOcrStrategy": "auto",
        "Accept": "text/plain",
    }

    # Retry loop
    attempts = 0
    while attempts < max_retries:
        try:
            async with session.post(xlsx_flask_url, data=form_data, headers=headers) as resp:
                status = resp.status
                try:
                    js = await resp.json()
                except:
                    js = await resp.text()

                if status == 429 or (500 <= status < 600):
                    # Retry on 429 or 5xx
                    log.warning(
                        f"[XLSX] Received status={status} from XLSX server. "
                        f"Retry {attempts+1}/{max_retries}..."
                    )
                    attempts += 1
                    await asyncio.sleep(2 * attempts)
                    continue

                # Otherwise, return whatever we got (200, 201, 4xx except 429, etc.)
                return (status, js)

        except Exception:
            # Log and retry
            log.exception(
                f"[XLSX] Exception while calling XLSX endpoint (attempt {attempts+1}/{max_retries}):"
            )
            attempts += 1
            await asyncio.sleep(2 * attempts)

    # If we get here, we've exceeded retries
    msg = f"[XLSX] Max retries ({max_retries}) exceeded while posting XLSX file."
    log.error(msg)
    return (500, msg)


def fetch_tika_bearer_token():
    """
    Fetch a Bearer token from Google Auth for Tika usage.
    Returns the token string, or raises an exception.
    """
    try:
        token_json = google_auth.impersonated_id_token(serverurl=SERVER_DOMAIN).json()
        return token_json["token"]
    except Exception as exc:
        log.exception("Failed to obtain impersonated ID token for Tika:")
        raise

async def handle_xlsx_blob_async(
    file_key,
    base_name,
    local_tmp_path,
    project_id,
    data_source_id,
    sub_for_hash,
    sub_id
):
    """
    Wraps your existing process_xlsx_blob call in an async context
    so it can be used with asyncio.gather.
    """
    # This corresponds to your "process_xlsx_blob" from your code
    # and returns (final_status, usage_credits, error_msg)
    try:
        return await process_xlsx_blob(
            file_key,
            base_name,
            local_tmp_path,
            project_id,
            data_source_id,
            sub_for_hash,
            sub_id,
        )
    except Exception as e:
        log.exception(f"[handle_xlsx_blob_async] XLSX processing error for {base_name}")
        return ("failed", 0, str(e))
    
async def process_xlsx_blob(
    file_key,
    base_name,
    local_tmp_path,
    project_id,
    data_source_id,
    sub_for_hash,
    sub_id,
):
    """
    High-level XLSX ingestion that opens a session and calls _call_xlsx_endpoint,
    returning final status/credits/error_msg.
    """
    final_status = "failed"
    usage_credits = 0
    error_msg = ""

    xlsx_flask_url = XLSX_SERVER_ENDPOINT

    # Create a brand new session each time we process a single XLSX
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=3600,      # Total timeout for entire operation
            sock_read=3600,  # Specifically set read timeout
            connect=30,      # Time to connect to server
            sock_connect=30, # Time to connect socket
        )
    ) as session:
        try:
            status_code, response_data = await _call_xlsx_endpoint(
                session,
                xlsx_flask_url,
                local_tmp_path,
                base_name,
                project_id,
                data_source_id,
                sub_for_hash,
                max_retries=5,
            )

            # Clean up local file if it exists
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

            if status_code == 200:
                final_status = "processed"
                if isinstance(response_data, dict):
                    usage_credits = response_data.get("chunks_processed", 0) * 1.5
                else:
                    usage_credits = 0
            else:
                error_msg = f"[XLSX] Ingest error: status={status_code}, resp={response_data}"

        except Exception:
            # Log full traceback
            log.exception("[XLSX] Exception calling XLSX ingestion:")
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
            error_msg = "Exception calling XLSX ingestion (see logs for traceback)."

    # session closes automatically here
    return (final_status, usage_credits, error_msg)

def finalize_data_source(data_source_id, ds, new_status="processed"):
    """
    Helper to finalize the data source after processing is done,
    setting lastSyncTime and nextSyncTime, etc.
    """
    now_dt = datetime.now(CENTRAL_TZ)
    next_sync = compute_next_sync_time(now_dt, ds.get("syncOption"))
    psql.update_data_source_by_id(
        data_source_id,
        status=new_status,
        lastSyncTime=now_dt.isoformat(),
        nextSyncTime=next_sync.isoformat(),
    )

def determine_file_extension(filename, default_ext="pdf"):
    """
    Determine the file extension by looking at the filename.
    If none found, return `default_ext`.
    """
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return default_ext


def process_local_file(local_tmp_path, extension):
    """
    Convert/transform the file at local_tmp_path into a list of (pageNum, io.BytesIO) pages.
    This function handles PDF, docx->PDF conversion, images as single-page, email formats, ODS, etc.
    
    Returns: (final_pages, final_num_pages)
    """
    # For PDF
    if extension == "pdf":
        with open(local_tmp_path, "rb") as f_in:
            pdf_data = f_in.read()
        return file_processor.split_pdf(pdf_data)

    # Convert to PDF from docx/pptx/etc.
    elif extension in [
        "docx", "odt", "odp", "odg", "odf", "fodt", "fodp", "fodg",
        "123", "dbf", "scm", "dotx", "docm", "dotm", "xml", "doc",
        "qpw", "pptx", "ppsx", "ppmx", "potx", "pptm", "ppam", "ppsm",
        "ppt", "pps", "ppa", "rtf"
    ]:
        pdf_data = convert_to_pdf(local_tmp_path, extension)
        if pdf_data:
            return file_processor.split_pdf(pdf_data)
        else:
            raise ValueError(f"Conversion to PDF failed for extension: {extension}")

    # Images
    elif extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
        with open(local_tmp_path, "rb") as f:
            image_data = f.read()
        pages = [("1", io.BytesIO(image_data))]
        return pages, len(pages)

    # Email formats
    elif extension in ["eml", "msg", "pst", "ost", "mbox", "dbx", "dat", "emlx"]:
        with open(local_tmp_path, "rb") as f:
            msg_data = f.read()
        pages = [("1", io.BytesIO(msg_data))]
        return pages, len(pages)

    # ODS
    elif extension == "ods":
        with open(local_tmp_path, "rb") as f:
            ods_data = f.read()
        pages = file_processor.split_ods(ods_data)
        return pages, len(pages)

    # If none match
    else:
        raise ValueError(f"Unsupported file extension: {extension}")


def process_file_with_tika(
    file_key,
    base_name,
    local_tmp_path,
    extension,
    project_id,
    data_source_id,
    last_modified,
    source_type,
    sub_id
):
    """
    Complete flow:
      1. Convert/split local file into pages.
      2. Async gather:
         - Tika extraction + embedding
         - /metadata call
      3. If metadata is non-empty, store in DB and attach to chunk metadata
      4. Return (final_status, number_of_pages, error_msg).
    """
    now_dt_str = datetime.now(CENTRAL_TZ).isoformat()

    try:
        final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
    except Exception as e:
        log.error(f"[process_file_with_tika] Failed to process/convert {base_name}: {str(e)}")
        return ("failed", 0, str(e))

    # We'll remove the local file (PDF or original) after we've read it
    # so we can still call fetch_additional_metadata with the same local_tmp_path.
    # If your new metadata API also processes PDF conversions, decide if you
    # want to call it BEFORE or AFTER the docx->pdf conversion, etc.
    # For simplicity,  we keep local_tmp_path as-is.

    # Bearer token for Tika
    try:
        bearer_token = fetch_tika_bearer_token()
    except Exception as e:
        # Clean up local file if desired
        return ("failed", 0, f"Failed to fetch Tika token: {str(e)}")

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "X-Tika-PDFOcrStrategy": "auto",
        "Accept": "text/plain",
    }

    # Create our async tasks
    async def run_parallel_tasks():
        async with aiohttp.ClientSession() as session:
            # Tika extraction
            extraction_task = asyncio.create_task(
                process_pages_async(
                    final_pages,
                    headers,
                    base_name,
                    project_id,
                    file_key,
                    data_source_id,
                    last_modified,
                    sourceType=source_type,
                )
            )
            # Metadata fetching
            metadata_task = asyncio.create_task(
                fetch_additional_metadata(
                    local_tmp_path,
                    file_key,
                    project_id,
                    sub_id or "no_subscription",
                )
            )
            # Wait for both to finish
            extracted_pages, metadata_json = await asyncio.gather(extraction_task, metadata_task)
            return extracted_pages, metadata_json

    loop = get_event_loop()
    try:
        results, metadata_json = loop.run_until_complete(run_parallel_tasks())
    except Exception as e:
        log.exception(f"[process_file_with_tika] Tika/metadata flow failed for {base_name}:")
        return ("failed", 0, str(e))
    finally:
        # Clean up the local file
        if os.path.exists(local_tmp_path):
            os.remove(local_tmp_path)

    # results is the list of (extracted_text, page_num) from Tika.
    # metadata_json is whatever came from /metadata.

    # If everything is successful, mark file processed
    psql.update_file_by_id(
        file_key,
        status="processed",
        pageCount=len(results),
        updatedAt=now_dt_str
    )

    # If we have metadata, store it in the DB
    if metadata_json:
        try:
            # Save the entire JSON to the "file" table (assuming you have such a column)
            psql.update_file_generated_metadata(file_key, metadata_json)
        except Exception:
            log.exception(f"[process_file_with_tika] Failed to store generatedMetadata in DB for file_id={file_key}")

    # Publish usage
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
            log.warning(f"[process_file_with_tika] Publish usage failed for {base_name}: {str(e)}")

    # Return success
    return ("processed", len(results), "")


def shutdown_handler(sig, frame):
    log.info(f"Caught signal {signal.strsignal(sig)}. Exiting.")
    sys.exit(0)
