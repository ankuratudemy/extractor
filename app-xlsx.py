#!/usr/bin/env python3
import os
import re
import math
import json
import hashlib
import logging
import traceback
import asyncio
from datetime import datetime

from flask import Flask, request, jsonify
import pandas as pd
import tiktoken
from flask_cors import CORS

# Pinecone gRPC wrapper or standard library
from pinecone.grpc import PineconeGRPC as Pinecone

# Vertex AI
from vertexai.preview.language_models import TextEmbeddingModel
from google import genai

# Logging + BM25
from shared.logging_config import log
from shared import psql
from shared.bm25 import (
    tokenize_document,
    publish_partial_bm25_update,
    compute_bm25_sparse_vector,
    get_project_vocab_stats,
)

# ------------------------------------------------------------------
# ENV / CONFIG
# ------------------------------------------------------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
VERTEX_MODEL_NAME = os.environ.get("VERTEX_MODEL_NAME", "text-multilingual-embedding-preview-0409")

ENVIRONMENT = os.environ.get("ENVIRONMENT", "prod")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# Pinecone & Vertex AI Setup
# ------------------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

google_genai_client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location="us-central1")

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)


# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_traceback(e):
    return "".join(traceback.format_exception(type(e), e, e.__traceback__))

def generate_md5_hash(*args):
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    return hashlib.md5(combined_string.encode("utf-8")).hexdigest()

def get_google_embedding_sync(texts, model_name=VERTEX_MODEL_NAME):
    """
    Synchronous call for embeddings; we'll wrap in async so it won't block.
    """
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=texts, auto_truncate=True)
    return [emb.values for emb in embeddings_list]

async def get_google_embedding(texts, model_name=VERTEX_MODEL_NAME):
    """
    Asynchronous wrapper. We'll do a threadpool run so it doesn't block event loop.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_google_embedding_sync, texts, model_name)

def chunk_text(text, max_tokens=2048, overlap_chars=2000):
    """
    Splits text into overlapping ~max_tokens chunks. Each chunk ~2k tokens with ~2k chars overlap.
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

        # Overlap
        overlap_str = chunk_str[-overlap_chars:] if len(chunk_str) > overlap_chars else chunk_str
        overlap_tokens = enc.encode(overlap_str)
        overlap_count = len(overlap_tokens)

        start = end - overlap_count
        end = start + max_tokens

    return chunks

def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int = 300) -> dict:
    """
    1) Combine the text
    2) publish partial BM25
    3) fetch vocab stats
    4) compute one sparse vector
    """
    combined_text = " ".join(file_texts)
    tokens = tokenize_document(combined_text)
    publish_partial_bm25_update(project_id, tokens, is_new_doc=True)
    vocab_stats = get_project_vocab_stats(project_id)
    return compute_bm25_sparse_vector(tokens, project_id, vocab_stats, max_terms=max_terms)

# ------------------------------------------------------------------
# ASYNC BATCH FUNCTIONS
# ------------------------------------------------------------------
async def create_and_upload_embeddings_in_batches(
    results, filename, namespace, file_id, last_modified, data_source_id=None
):
    """
    results: list of (text, sub_index) for the entire sheet
    1) gather all text => single BM25 update => get sparse_values
    2) chunk, embed, upsert in batches
    """
    all_texts = [text for (text, _idx) in results if text.strip()]
    if not all_texts:
        log.info("No text to process in create_and_upload_embeddings_in_batches.")
        return

    # 1) Single-file BM25 update
    sparse_values = get_sparse_vector(all_texts, namespace, 300)

    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    total_chunks = 0
    for text, sub_idx in results:
        if not text.strip():
            continue

        text_chunks = chunk_text(text, max_tokens=2048, overlap_chars=2000)
        for i, chunk in enumerate(text_chunks):
            chunk_token_len = len(enc.encode(chunk))

            if ((batch_text_count + 1) > max_batch_texts) or (batch_token_count + chunk_token_len > max_batch_tokens):
                if batch:
                    await process_embedding_batch(
                        batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
                    )
                    total_chunks += len(batch)
                    batch.clear()
                    batch_text_count = 0
                    batch_token_count = 0

            if ((batch_text_count + 1) <= max_batch_texts) and ((batch_token_count + chunk_token_len) <= max_batch_tokens):
                batch.append((chunk, sub_idx, i))
                batch_text_count += 1
                batch_token_count += chunk_token_len
            else:
                log.warning("Chunk too large or logic error preventing batch addition.")
                continue

    # Process any remainder
    if batch:
        await process_embedding_batch(
            batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
        )
        total_chunks += len(batch)
        batch.clear()

    log.info(f"Total text chunks processed/upserted: {total_chunks}")

async def process_embedding_batch(
    batch, filename, namespace, file_id, data_source_id, last_modified, sparse_values
):
    texts = [item[0] for item in batch]
    embeddings = await get_google_embedding(texts)
    vectors = []

    for (chunk_text, sub_idx, chunk_idx), embedding in zip(batch, embeddings):
        if data_source_id:
            doc_id = f"{data_source_id}#{file_id}#{sub_idx}#{chunk_idx}"
        else:
             doc_id = f"{file_id}#{sub_idx}#{chunk_idx}"
        metadata = {
            "text": chunk_text,
            "source": filename,
            "page": sub_idx,
            "sourceType": "excelSheet",  # you might want to rename this to 'spreadsheet'
            "fileId": file_id,
            "lastModified": (
                last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified)
            ),
            **({"dataSourceId": data_source_id} if data_source_id is not None else {})
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
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        log.info(f"Upserted {len(vectors)} vectors to Pinecone.")
    except Exception as e:
        log.error(f"Failed upserting vectors to Pinecone for file_id={file_id}: {e}")


# ------------------------------------------------------------------
# SHEET PROCESSING W/ LLM HEADERS
# ------------------------------------------------------------------
def clean_sheet_name(sheet_name):
    """Replace non-alphanumeric chars with underscores."""
    cleaned_name = re.sub(r"[^a-zA-Z0-9]", "_", sheet_name)
    cleaned_name = cleaned_name.strip("_")
    cleaned_name = re.sub(r"_+", "_", cleaned_name)
    return cleaned_name

def process_sheet_with_header_llm(sheet_name, df):
    """
    1) Take up to 50 sample rows => produce 'header' text via LLM
    2) Then chunk the entire sheet (all rows) in ~2000 token blocks, each prefixed with the header text
    3) Return list of (chunk_text, chunk_index)
    """
    try:
        # 1) Build a 'sample' representation
        sample_rows = min(math.ceil(len(df) * 0.1), 50)
        sample_df = df.head(sample_rows)
        sample_data_md = sample_df.to_markdown(index=False)  # markdown style

        # System & question to LLM:
        system_instruction = (
            "You are a precise AI agent. Your task is to extract the header columns and any general info "
            "above the headers in markdown format as a string from the sample Excel (or CSV) data."
        )
        question = f"EXCEL/CSV SAMPLE ROWS:\n\n{sample_data_md}\n\nJust produce the header columns + any data above header."

        extracted_header = google_genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=question,
            config={"system_instruction": system_instruction, "temperature": 0.3},
        )
        header_text = extracted_header.text

        # 2) We'll create big ~2000 token chunks. Each chunk includes the `header_text` plus a batch of rows.
        enc = tiktoken.get_encoding("cl100k_base")
        header_tokens = len(enc.encode(header_text))
        max_tokens = 2000

        # We'll collect row data in 'blocks' that, with header_text, stay < max_tokens
        chunks = []
        current_block_rows = []
        current_block_token_count = header_tokens

        for row_i, row in df.iterrows():
            row_data = " | ".join(str(item) for item in row)
            row_tokens = len(enc.encode(row_data))

            if current_block_token_count + row_tokens >= max_tokens:
                # push the block
                block_str = header_text + "\n" + "\n".join(current_block_rows)
                chunks.append(block_str)
                # reset
                current_block_rows = []
                current_block_token_count = header_tokens

            current_block_rows.append(row_data)
            current_block_token_count += row_tokens

        # last block
        if current_block_rows:
            block_str = header_text + "\n" + "\n".join(current_block_rows)
            chunks.append(block_str)

        # Return an array of (chunk_text, chunk_idx)
        results = [(chunk_str, idx) for idx, chunk_str in enumerate(chunks)]
        return results

    except Exception as e:
        logging.error(f"Error in process_sheet_with_header_llm({sheet_name}): {e}")
        logging.error(get_traceback(e))
        return []

def load_spreadsheet_as_dict(file_path):
    """
    1) Detect extension.
    2) For CSV/TSV => load single DataFrame and store in a dict with "Sheet1".
    3) For Excel-like files => pd.read_excel(sheet_name=None).
    4) For .ots (ODF spreadsheet) => also read with the 'odf' engine if installed.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(file_path, sep=sep, na_values="Missing")
        return {"Sheet1": df}
    elif ext in [".xls", ".xltm", ".xltx", ".xlsx"]:
        # standard Excel engine
        all_sheets = pd.read_excel(file_path, sheet_name=None, na_values="Missing")
        return all_sheets
    elif ext == ".ots":
        # requires installing 'odfpy' or 'openpyxl' with ODF support
        # engine="odf" might be required for OTS/ODS
        all_sheets = pd.read_excel(file_path, sheet_name=None, na_values="Missing", engine="odf")
        return all_sheets
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def extract_spreadsheet_into_chunks(file_path):
    """
    Reads the spreadsheet from `file_path`,
    returns { sheet_name: [(chunk_text, idx), ...], ... }
    (mimicking your original structure).
    """
    sheet_data = {}
    try:
        all_sheets = load_spreadsheet_as_dict(file_path)
        for raw_name, df in all_sheets.items():
            sname = clean_sheet_name(raw_name)
            results = process_sheet_with_header_llm(sname, df)
            sheet_data[sname] = results
    except Exception as e:
        logging.error(f"Error in extract_spreadsheet_into_chunks: {e}")
        logging.error(get_traceback(e))
    return sheet_data

# ------------------------------------------------------------------
# MAIN INGEST FUNCTION
# ------------------------------------------------------------------
def process_spreadsheet_file(uploaded_file, project_id, data_source_id, sub_id):
    """
    1) Save to /tmp
    2) For each sheet => produce chunked text (with LLM header)
    3) For each sheet => call 'create_and_upload_embeddings_in_batches' (async)
    4) Return overall stats
    """
    tmp_path = os.path.join("/tmp", uploaded_file.filename)
    uploaded_file.save(tmp_path)
    if data_source_id:
        file_id = generate_md5_hash(sub_id, project_id, data_source_id, uploaded_file.filename)
    else:
        file_id = generate_md5_hash(sub_id, project_id, uploaded_file.filename)

    # Extract chunks per sheet
    all_sheets_data = extract_spreadsheet_into_chunks(tmp_path)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    last_modified = datetime.now()

    total_sheets = 0
    total_chunks = 0

    for sheet_name, results_list in all_sheets_data.items():
        if not results_list:
            continue

        # Each 'results_list' is [ (chunk_text, chunk_idx), ... ]
        asyncio.run(
            create_and_upload_embeddings_in_batches(
                results=results_list,
                filename=f"{uploaded_file.filename}::{sheet_name}",
                namespace=project_id,
                file_id=file_id,
                last_modified=last_modified,
                **({"data_source_id": data_source_id} if data_source_id is not None else {})
            )
        )
        total_sheets += 1
        total_chunks += len(results_list)

    return {
        "status": "success",
        "message": f"Processed file '{uploaded_file.filename}' with LLM-based header chunking.",
        "sheets_processed": total_sheets,
        "chunks_processed": total_chunks,
    }


# ------------------------------------------------------------------
# FLASK ENDPOINT
# ------------------------------------------------------------------
@app.route("/process_spreadsheet", methods=["POST"])
def process_spreadsheet():
    """
    Expects a multipart/form-data POST with:
      - 'file' => the spreadsheet-like file (csv, xlsx, etc.)
      - 'project_id'
      - 'data_source_id' (optional)
      - 'sub_id'
    """
    logging.info(f"Spreadsheet REQUEST# {request.files}")
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    uploaded_file = request.files["file"]

    project_id = request.form.get("project_id")
    data_source_id = request.form.get("data_source_id", None)
    sub_id = request.form.get("sub_id")

    if not project_id or not sub_id:
        return jsonify({"error": "Missing project_id or sub_id"}), 400

    try:
        result = process_spreadsheet_file(uploaded_file, project_id, data_source_id, sub_id)
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error processing spreadsheet: {str(e)}")
        logging.error(get_traceback(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # In production, run via gunicorn or uwsgi
    app.run(host="0.0.0.0", port=9999)
