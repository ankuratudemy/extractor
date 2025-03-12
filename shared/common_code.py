import os
import re
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

# ---------------------
# Import the generate() function for topics (batch version now)
# ---------------------
from google import genai
from google.genai import types

# ----------------------------------------------------------------------------
# ENV VARIABLES
# ----------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER", "/tmp/uploads")  # default
METADATA_SERVER_URL = os.environ.get("METADATA_SERVER_URL", None)
METADATA_ENDPOINT = (
    f"https://{METADATA_SERVER_URL}/get_metadata" if METADATA_SERVER_URL else None
)

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


# ----------------------------------------------------------------------------
# NEW: Function to batch multiple chunks together for single LLM call
# ----------------------------------------------------------------------------
async def generate_topics_for_batch(chunks: List[str], topics_prompt: str, retries: int = 3):
    """
    Combine multiple text chunks into one LLM request to reduce overhead.
    We ask the model to return a JSON *array of arrays*, each sub-array holding
    the topics for the corresponding chunk index.
    """
    regions = [
        "us-central1",
        "us-east5",   # Columbus, Ohio
        "us-south1",  # Dallas, Texas
        "us-west4",   # Las Vegas, Nevada
        "us-east1",   # Moncks Corner, SC
        "us-east4",   # Northern Virginia
        "us-west1"    # Oregon
    ]

    # Construct a prompt that instructs the model to output strictly valid JSON:
    # an array of arrays, each sub-array is for the chunk at that index.
    prompt_text = f"""{topics_prompt}

We have {len(chunks)} distinct text chunks. For each chunk, produce a list of 5-100 important keywords or phrases, including variations like abbreviations, expansions, etc. 
Return strictly valid JSON: an array of length={len(chunks)}, where element i is a (string) list of topics for chunk i.

Chunks:
{json.dumps(chunks, ensure_ascii=False)}

Return only the JSON.
"""

    text_part = types.Part.from_text(text=prompt_text)
    model = "gemini-2.0-flash-lite-001"

    contents = [types.Content(role="user", parts=[text_part])]

    # Important: response_schema indicates we expect a list-of-lists-of-strings
    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=0.95,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        response_mime_type="application/json",
        response_schema={
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
    )

    for region in regions:
        client = genai.Client(
            vertexai=True,
            project=os.environ.get("GCP_PROJECT_ID"),
            location=region,
        )

        attempt = 0
        while attempt < retries:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                )
                # The response should be a JSON array of arrays if successful
                text_result = response.text
                try:
                    parsed = json.loads(text_result)
                    # parsed should be a list of lists
                    return parsed
                except json.JSONDecodeError:
                    log.error(f"[Topics] Could not parse LLM output as JSON in region={region}.")
                    # Fallback: return empty array-of-arrays
                    return [[] for _ in chunks]

            except Exception as e:
                error_message = str(e)
                print(f"Attempt {attempt + 1} in region {region} failed: {error_message}")
                if "429" in error_message:
                    print(f"Region {region} rate limited, switching to next region.")
                    break  # Break inner loop, move to next region
                attempt += 1
                await asyncio.sleep(2 ** attempt)

    log.error("[Topics] All region attempts failed. Returning empty arrays.")
    return [[] for _ in chunks]


async def fetch_additional_metadata(
    file_path: str, file_id: str, project_id: str, sub_id: str
) -> dict:
    global METADATA_SESSION

    if not METADATA_ENDPOINT:
        log.info("[Metadata] METADATA_SERVER_URL not set. Skipping metadata fetch.")
        return {}

    if METADATA_SESSION is None or METADATA_SESSION.closed:
        timeout = aiohttp.ClientTimeout(
            total=600, connect=30, sock_connect=30, sock_read=600
        )
        METADATA_SESSION = aiohttp.ClientSession(timeout=timeout)

    if not os.path.exists(file_path):
        log.warning(f"[Metadata] File {file_path} does not exist. Cannot fetch metadata.")
        return {}

    with open(file_path, "rb") as f:
        file_content = f.read()

    try:
        metadata_bearer_token = google_auth.impersonated_id_token(
            serverurl=METADATA_SERVER_URL
        ).json()["token"]
    except Exception:
        log.exception("[Metadata] Failed to get impersonated token for metadata server:")
        return {}

    headers = {
        "Authorization": f"Bearer {metadata_bearer_token}",
        "Accept": "application/json",
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # IMPORTANT: Re‑create FormData each time before calling `post()`
            form_data = aiohttp.FormData()
            form_data.add_field("file", file_content, filename=os.path.basename(file_path))
            form_data.add_field("project_id", str(project_id))
            form_data.add_field("sub_id", str(sub_id))

            async with METADATA_SESSION.post(
                METADATA_ENDPOINT, data=form_data, headers=headers
            ) as resp:
                if resp.status == 429 or (500 <= resp.status < 600):
                    log.warning(
                        f"[Metadata] status={resp.status} from /metadata. "
                        f"Retry {attempt+1}/{max_retries}."
                    )
                    await asyncio.sleep(2 * (attempt + 1))
                    continue

                if resp.status >= 400:
                    log.error(f"[Metadata] Non-success HTTP status: {resp.status}")
                    return {}

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

    log.error("[Metadata] Max retries exceeded for /metadata call.")
    return {}


def parse_last_sync_time(last_sync_time_value):
    if isinstance(last_sync_time_value, datetime):
        return ensure_timezone_aware(last_sync_time_value)
    if isinstance(last_sync_time_value, str):
        try:
            return ensure_timezone_aware(parser.isoparse(last_sync_time_value))
        except:
            return None
    return None


def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int) -> dict:
    combined_text = " ".join(file_texts)
    tokens = tokenize_document(combined_text)
    publish_partial_bm25_update(project_id, tokens, is_new_doc=True)
    vocab_stats = get_project_vocab_stats(project_id)
    return compute_bm25_sparse_vector(
        tokens, project_id, vocab_stats, max_terms=max_terms
    )


def ensure_timezone_aware(dt):
    CENTRAL_TZ = ZoneInfo("America/Chicago")
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
    CENTRAL_TZ = ZoneInfo("America/Chicago")
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


def remove_missing_files(
    db_file_keys, remote_file_keys, data_source_id, project_id, namespace=None
):
    removed_keys = db_file_keys - remote_file_keys
    log.info(f"Removed keys: {removed_keys}")
    for r_key in removed_keys:
        remove_file_from_db_and_pinecone(
            r_key, data_source_id, project_id, namespace=namespace or project_id
        )


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
    except subprocess.CalledProcessError:
        log.exception("Conversion to PDF failed:")
        return None
    except Exception:
        log.exception("Error during PDF conversion:")
        return None


async def _async_put_page(session, url, page_data, page_num, headers, max_retries=5):
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
                    log.warning(
                        f"[Tika] Received status={resp.status} for page={page_num}. "
                        f"Retry {retries+1}/{max_retries}..."
                    )
                    retries += 1
                    await asyncio.sleep(2 * retries)
                    continue
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

    msg = f"[Tika] Failed to process page {page_num} after {max_retries} retries."
    log.error(msg)
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
    additional_metadata=None,
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
        additional_metadata=additional_metadata,
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


# ----------------------------------------------------------------------------
# NEW: Helper for sub-batching lists in groups of 'size=50'
# ----------------------------------------------------------------------------
def chunk_list(data, size=50):
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def process_embedding_batch(
    batch,
    filename,
    namespace,
    file_id,
    data_source_id,
    last_modified,
    sparse_values,
    sourceType,
    additional_metadata=None,
):
    """
    Sends a batch of chunked text to the embedding model, then calls the
    LLM for topics in sub-batches of size=50. Then upserts to Pinecone.
    """
    texts = [item[0] for item in batch]

    # 1) Embeddings for all chunked text:
    embeddings = await get_google_embedding(texts)

    # 2) Generate topics in sub-batches to avoid token-limit blowups.
    TOPICS_PROMPT = (
        "From the following text chunks, extract the most important keywords or key phrases "
        "(5-100 terms). Also include variations, abbreviations, expansions, etc. "
        "Return strictly valid JSON as an array-of-arrays, each sub-array for one chunk."
    )
    all_topics = []

    # We'll gather topic results for each chunk in the same order
    text_index = 0
    while text_index < len(texts):
        sub_batch_texts = texts[text_index : text_index + 50]  # up to 50
        sub_topics = await generate_topics_for_batch(sub_batch_texts, TOPICS_PROMPT)
        all_topics.extend(sub_topics)
        text_index += 50

    # 3) Build vectors and upsert
    vectors = []
    for ((text, page_num, chunk_idx), embedding, parsed_topics) in zip(batch, embeddings, all_topics):
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
        }
        if data_source_id is not None:
            metadata["dataSourceId"] = data_source_id

        # Attach any additional metadata fields
        if additional_metadata and "metadataTags" in additional_metadata:
            metadata.update(additional_metadata["metadataTags"])

        # Attach parsed topics
        metadata["auto_topics"] = parsed_topics

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


async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    return [emb.values for emb in embeddings_list]


async def create_and_upload_embeddings_in_batches(
    results,
    filename,
    namespace,
    file_id,
    data_source_id,
    last_modified,
    sourceType,
    additional_metadata,
):
    """
    Takes raw extracted text from all pages, splits each text into chunks,
    and sends them (in manageable batch sizes) to process_embedding_batch.
    Also calls generate_topics_for_batch in sub-batches of 50.
    """
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    # Single-file BM25 update
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
            additional_metadata,
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
    with open(file_path, "rb") as f:
        file_content = f.read()

    try:
        xlsx_bearer_token = google_auth.impersonated_id_token(
            serverurl=XLSX_SERVER_URL
        ).json()["token"]
    except Exception:
        log.exception("[XLSX] Failed to obtain impersonated ID token:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        return (500, "Failed to obtain impersonated ID token.")

    headers = {
        "Authorization": f"Bearer {xlsx_bearer_token}",
        "X-Tika-PDFOcrStrategy": "auto",
        "Accept": "text/plain",
    }

    attempts = 0
    while attempts < max_retries:
        try:
            form_data = aiohttp.FormData()
            form_data.add_field(
                "file",
                file_content,
                filename=base_name,
            )
            form_data.add_field("project_id", project_id)
            form_data.add_field("sub_id", sub_for_hash)
            if data_source_id is not None:
                form_data.add_field("dataSourceId", data_source_id)

            async with session.post(
                xlsx_flask_url, data=form_data, headers=headers
            ) as resp:
                status = resp.status
                try:
                    js = await resp.json()
                except:
                    js = await resp.text()

                if status == 429 or (500 <= status < 600):
                    log.warning(
                        f"[XLSX] status={status} from server. Retry {attempts+1}/{max_retries}..."
                    )
                    attempts += 1
                    await asyncio.sleep(2 * attempts)
                    continue

                return (status, js)

        except Exception:
            log.exception(f"[XLSX] Exception calling XLSX endpoint (attempt {attempts+1}/{max_retries}):")
            attempts += 1
            await asyncio.sleep(2 * attempts)

    msg = f"[XLSX] Max retries ({max_retries}) exceeded while posting XLSX file."
    log.error(msg)
    return (500, msg)


def fetch_tika_bearer_token():
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
    sub_id,
):
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
    final_status = "failed"
    usage_credits = 0
    error_msg = ""

    xlsx_flask_url = XLSX_SERVER_ENDPOINT

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(
            total=3600,
            sock_read=3600,
            connect=30,
            sock_connect=30,
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
            log.exception("[XLSX] Exception calling XLSX ingestion:")
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
            error_msg = "Exception calling XLSX ingestion (see logs)."

    return (final_status, usage_credits, error_msg)


def finalize_data_source(data_source_id, ds, new_status="processed"):
    now_dt = datetime.now(CENTRAL_TZ)
    next_sync = compute_next_sync_time(now_dt, ds.get("syncOption"))
    psql.update_data_source_by_id(
        data_source_id,
        status=new_status,
        lastSyncTime=now_dt.isoformat(),
        nextSyncTime=next_sync.isoformat(),
    )


def determine_file_extension(filename, default_ext="pdf"):
    if "." in filename:
        return filename.rsplit(".", 1)[-1].lower()
    return default_ext


def process_local_file(local_tmp_path, extension):
    if extension == "pdf":
        with open(local_tmp_path, "rb") as f_in:
            pdf_data = f_in.read()
        return file_processor.split_pdf(pdf_data)

    elif extension in [
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
        "ppt",
        "pps",
        "ppa",
        "rtf",
    ]:
        pdf_data = convert_to_pdf(local_tmp_path, extension)
        if pdf_data:
            return file_processor.split_pdf(pdf_data)
        else:
            raise ValueError(f"Conversion to PDF failed for extension: {extension}")

    elif extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp"]:
        with open(local_tmp_path, "rb") as f:
            image_data = f.read()
        pages = [("1", io.BytesIO(image_data))]
        return pages, len(pages)

    elif extension in ["eml", "msg", "pst", "ost", "mbox", "dbx", "dat", "emlx"]:
        with open(local_tmp_path, "rb") as f:
            msg_data = f.read()
        pages = [("1", io.BytesIO(msg_data))]
        return pages, len(pages)

    elif extension == ["ods"]:
        with open(local_tmp_path, "rb") as f:
            ods_data = f.read()
        pages = file_processor.split_ods(ods_data)
        return pages, len(pages)

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
    sub_id,
):
    now_dt_str = datetime.now(CENTRAL_TZ).isoformat()

    try:
        final_pages, final_num_pages = process_local_file(local_tmp_path, extension)
    except Exception as e:
        log.error(f"[process_file_with_tika] Failed to process/convert {base_name}: {str(e)}")
        return ("failed", 0, str(e))

    try:
        bearer_token = fetch_tika_bearer_token()
    except Exception as e:
        return ("failed", 0, f"Failed to fetch Tika token: {str(e)}")

    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "X-Tika-PDFOcrStrategy": "auto",
        "Accept": "text/plain",
    }

    loop = get_event_loop()

    async def run_tasks_sequential():
        metadata_json = await fetch_additional_metadata(
            file_path=local_tmp_path,
            file_id=file_key,
            project_id=project_id,
            sub_id=sub_id or "no_subscription",
        )

        extracted_pages = await process_pages_async(
            pages=final_pages,
            headers=headers,
            filename=base_name,
            namespace=project_id,
            file_id=file_key,
            data_source_id=data_source_id,
            last_modified=last_modified,
            sourceType=source_type,
            additional_metadata=metadata_json,
        )
        return extracted_pages, metadata_json

    try:
        results, metadata_json = loop.run_until_complete(run_tasks_sequential())
    except Exception as e:
        log.exception(f"[process_file_with_tika] Tika/metadata flow failed for {base_name}:")
        return ("failed", 0, str(e))
    finally:
        if os.path.exists(local_tmp_path):
            os.remove(local_tmp_path)

    # results => list of (extracted_text, page_num)
    psql.update_file_by_id(
        file_key, status="processed", pageCount=len(results), updatedAt=now_dt_str
    )

    if metadata_json and "metadataTags" in metadata_json:
        try:
            psql.update_file_generated_metadata(file_key, metadata_json["metadataTags"])
        except Exception:
            log.exception(
                f"[process_file_with_tika] Failed to store generatedMetadata for file_id={file_key}"
            )

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
            log.warning(f"[process_file_with_tika] Publish usage failed: {str(e)}")

    return ("processed", len(results), "")


def shutdown_handler(sig, frame):
    log.info(f"Caught signal {signal.strsignal(sig)}. Exiting.")
    sys.exit(0)
