from shared.logging_config import log
import os
import io
from flask import Flask, request
from google.cloud import storage
from vertexai.preview.language_models import TextEmbeddingModel
from werkzeug.utils import secure_filename
from shared import file_processor, google_auth, security, google_pub_sub, psql
import json
import asyncio
import sys
import requests
import aiohttp
import tiktoken
import hashlib
from datetime import datetime
import signal
import ssl
from typing import List, Tuple
from pinecone import Pinecone

sys.path.append('../')

app = Flask(__name__)
app.debug = True

# Environment variables
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_CREDIT_USAGE_TOPIC = os.environ.get('GCP_CREDIT_USAGE_TOPIC')
UPLOADS_FOLDER = os.environ.get('UPLOADS_FOLDER')
SERVER_URL = f"https://{os.environ.get('SERVER_URL')}/tika"

app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER

# Initialize connection to Pinecone
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX_NAME')
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def generate_md5_hash(*args):
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = '|'.join(serialized_args)
    md5_hash = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
    return md5_hash


def download_file(bucket_name, filename, temp_file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    blob.download_to_filename(temp_file_path)
    log.info(
        f'Downloaded {filename} from bucket {bucket_name} to {temp_file_path}')
    return blob.content_type


def convert_to_pdf(file_path, file_extension):
    import subprocess
    try:
        pdf_file_path = os.path.splitext(file_path)[0] + '.pdf'
        command = [
            '/opt/libreoffice7.6/program/soffice',
            '--headless',
            '--convert-to',
            'pdf:writer_pdf_Export:{"SelectPdfVersion":{"type":"long","value":"17"}, "UseTaggedPDF": {"type":"boolean","value":"true"}}',
            '--outdir',
            os.path.dirname(file_path),
            file_path,
        ]
        subprocess.run(command, check=True)
        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, 'rb') as f:
                pdf_data = f.read()
            os.remove(pdf_file_path)
            return pdf_data
        else:
            return None
    except subprocess.CalledProcessError as e:
        log.error(f'Conversion to PDF failed: {str(e)}')
        return None
    except Exception as e:
        log.error(f'Error during PDF conversion: {str(e)}')
        return None


async def get_google_embedding(queries):
    embedder_name = "text-multilingual-embedding-preview-0409"
    model = TextEmbeddingModel.from_pretrained(embedder_name)
    log.info(f"Getting embeddings for {len(queries)} queries.")
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    embeddings = [embedding.values for embedding in embeddings_list]
    log.info("Embeddings fetched successfully.")
    return embeddings


async def async_put_request(session, url, payload, page_num, headers, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            payload_copy = io.BytesIO(payload.getvalue())
            async with session.put(url, data=payload_copy, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 429:
                    log.warning(
                        f"Retrying request for page {page_num}, Retry #{retries + 1}")
                    retries += 1
                    await asyncio.sleep(1)
                    continue
                content = await response.read()
                text_content = content.decode('utf-8', errors='ignore')
                log.info(
                    f"Page {page_num} processed successfully with length {len(text_content)} chars.")
                return text_content, page_num

        except (aiohttp.ClientError, ssl.SSLError, asyncio.TimeoutError) as e:
            log.error(
                f"Error during request for page {page_num}: {str(e)}, retrying...")
            retries += 1
            await asyncio.sleep(1)

    raise RuntimeError(
        f"Failed after {max_retries} retries for page {page_num}")


def upload_to_pinecone(vectors, namespace):
    log.info(f"Uploading {len(vectors)} vectors to Pinecone.")
    indexres = index.upsert(vectors=vectors, namespace=namespace)
    log.info(f"Upsert response: {indexres}")
    return indexres


async def process_embedding_batch(batch, filename, namespace, file_id):
    texts = [item[0] for item in batch]
    page_nums = [item[1] for item in batch]
    chunk_indexes = [item[2] for item in batch]

    log.info(f"Processing batch of size {len(batch)} for embeddings.")
    embeddings = await get_google_embedding(texts)

    vectors = []
    for (text, page_num, chunk_idx), embedding in zip(batch, embeddings):
        document_id = f"{file_id}#{page_num}#{chunk_idx}"
        metadata = {
            "text": text,
            "source": filename,
            "page": page_num
        }
        vectors.append({
            "id": document_id,
            "values": embedding,
            "metadata": metadata
        })

    upload_to_pinecone(vectors, namespace)
    # Clear memory
    del vectors, embeddings, texts, page_nums, chunk_indexes
    log.info("Batch processed and uploaded successfully.")


def chunk_text(text, max_tokens=2048, overlap_chars=2000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    token_count = len(tokens)
    log.info(
        f"Chunking text. Total tokens: {token_count}, max_tokens: {max_tokens}, overlap_chars: {overlap_chars}")

    if token_count <= max_tokens:
        log.info("No chunking needed, text fits in single chunk.")
        return [text]

    chunks = []
    start = 0
    end = max_tokens

    while start < token_count:
        chunk_tokens = tokens[start:end]
        chunk_text_str = enc.decode(chunk_tokens)
        chunk_len = len(chunk_tokens)
        chunks.append(chunk_text_str)
        log.info(
            f"Created chunk with {chunk_len} tokens from {start} to {end}.")

        if end >= token_count:
            log.info("Reached end of tokens.")
            break

        # Overlap logic
        overlap_str = chunk_text_str[-overlap_chars:] if len(
            chunk_text_str) > overlap_chars else chunk_text_str
        overlap_tokens = enc.encode(overlap_str)
        overlap_count = len(overlap_tokens)
        log.info(
            f"Overlap_count: {overlap_count} tokens for overlap_str length {len(overlap_str)} chars.")

        start = end - overlap_count
        end = start + max_tokens
        log.info(f"Next chunk start: {start}, end: {end}")

    log.info(f"Total chunks created: {len(chunks)}")
    for i, c in enumerate(chunks):
        c_len = len(enc.encode(c))
        log.info(f"Chunk {i+1}/{len(chunks)}: {c_len} tokens")

    return chunks


async def create_and_upload_embeddings_in_batches(results, filename, namespace, file_id):
    """
    For each request (batch), we have these constraints:
    - Max 250 texts per batch
    - Max 20,000 tokens total per batch (we restrict it to 14,000)
    - Each text is already chunked to <=2048 tokens by chunk_text
    """
    batch = []
    batch_token_count = 0
    batch_text_count = 0
    max_batch_texts = 250
    max_batch_tokens = 14000
    enc = tiktoken.get_encoding("cl100k_base")

    log.info("Creating and uploading embeddings with new constraints.")
    for text_content, page_num in results:
        if not text_content.strip():
            log.info(f"Skipping embedding for empty text in page {page_num}")
            continue

        content_len = len(text_content)
        log.info(
            f"Processing text from page {page_num}, length {content_len} chars.")
        text_chunks = chunk_text(
            text_content, max_tokens=2048, overlap_chars=2000)
        log.info(f"{len(text_chunks)} chunks created for page {page_num}.")

        for i, chunk in enumerate(text_chunks):
            chunk_tokens = enc.encode(chunk)
            chunk_token_len = len(chunk_tokens)
            log.info(
                f"Next chunk {i+1}/{len(text_chunks)} from page {page_num}. Chunk token length: {chunk_token_len}")

            # Check if adding this chunk would exceed batch limits
            if (batch_text_count + 1 > max_batch_texts) or (batch_token_count + chunk_token_len > max_batch_tokens):
                if batch:
                    log.info(f"Batch limits reached. Processing current batch. "
                             f"Batch size: {batch_text_count}, token count: {batch_token_count}.")
                    await process_embedding_batch(batch, filename, namespace, file_id)
                    batch.clear()
                    batch_token_count = 0
                    batch_text_count = 0
                else:
                    # This should not happen because chunk_text ensures <=2048 tokens
                    log.warning(
                        "Single chunk exceeds allowed tokens or texts. Check logic if this warning appears.")

            # Now we can add the chunk
            if (batch_text_count + 1 <= max_batch_texts) and (batch_token_count + chunk_token_len <= max_batch_tokens):
                batch.append((chunk, page_num, i))
                batch_token_count += chunk_token_len
                batch_text_count += 1
                log.info(
                    f"Added chunk to batch. Current batch texts: {batch_text_count}, token count: {batch_token_count}")
            else:
                log.error("Unable to add chunk even after clearing batch. "
                          "Chunk might be too large or logic error occurred.")
                return

    # Process any remaining items
    if batch:
        log.info(
            f"Processing remaining batch of size {batch_text_count}, token count {batch_token_count}.")
        await process_embedding_batch(batch, filename, namespace, file_id)


async def process_pages_async(pages, headers, filename, namespace, file_id):
    url = SERVER_URL
    log.info(f"Starting async processing of {len(pages)} pages.")
    async with aiohttp.ClientSession() as session:
        tasks = [async_put_request(
            session, url, page_data, page_num, headers) for page_num, page_data in pages]
        results = await asyncio.gather(*tasks)

    log.info("All pages processed. Now creating and uploading embeddings.")
    await create_and_upload_embeddings_in_batches(results, filename, namespace, file_id)
    return results


@app.route('/', methods=['POST'])
def event_handler():
    event_data = request.get_json()
    log.info(f"Event Data: {event_data}")
    if not event_data:
        return 'No event data', 400

    try:
        bucket_name = event_data.get('bucket')
        file_name = event_data.get('name')
    except (KeyError, AttributeError):
        return 'Missing required fields in event data', 400

    if bucket_name and file_name:
        folder_name, file_name_only = os.path.split(file_name)
        filename = file_name_only
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # ---------------------
        #  Extract folder_name parts
        # ---------------------
        log.info(f'Folder: {folder_name}, File: {file_name_only}')
        folder_name_parts = folder_name.split('/')
        if len(folder_name_parts) < 3:
            log.error('Invalid folder path format')
            return 'Invalid folder path format', 400

        subscription_id = folder_name_parts[0]
        project_id = folder_name_parts[1]
        user_id = folder_name_parts[2]
        log.info(
            f"Subscription ID: {subscription_id}, Project ID: {project_id}, User ID: {user_id}")

        # ---------------------
        #  Check remaining credits first!
        # ---------------------
        remaining_credits = psql.get_remaining_credits(subscription_id)
        log.info(
            f"Remaining credits for subscription {subscription_id}: {remaining_credits}")

        # If no remaining credits, short-circuit and update status
        file_id = generate_md5_hash(subscription_id, project_id, filename)
        if remaining_credits <= 0:
            log.error(
                f"No credits left for subscription {subscription_id}. File {filename} not processed.")
            psql.update_file_status(
                id=file_id, status="no credits", page_nums=0, updatedAt=datetime.now())
            return 'No credits left for subscription, skipping file processing', 402

        # We do have credits, proceed...
        content_type_header = download_file(
            bucket_name, file_name, temp_file_path)
        log.info(f"Generated file_id: {file_id}")
        psql.update_file_status(
            id=file_id, status="processing", page_nums=0, updatedAt=datetime.now())

        default_ocr_strategy = 'auto'
        default_out_format = 'text/plain'
        x_tika_ocr_language = ''
        x_tika_pdf_ocr_strategy = default_ocr_strategy
        x_tika_accept = default_out_format

        headers = {
            'X-Tika-PDFOcrStrategy': x_tika_pdf_ocr_strategy,
            'Accept': x_tika_accept
        }
        if x_tika_ocr_language:
            headers['X-Tika-OCRLanguage'] = x_tika_ocr_language

        log.info(f"Content-Type: {content_type_header}")
        contentType = "application/pdf"

        oFileExtMap = {
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
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template": 'dotx',
            "application/vnd.ms-word.document.macroenabled.12": 'docm',
            "application/vnd.ms-word.template.macroenabled.12": 'dotm',
            "application/xml": 'xml',
            "application/msword": 'doc',
            "application/rtf": 'rtf',
            "application/wordperfect": 'wpd',
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 'xlsx',
            "application/vnd.openxmlformats-officedocument.spreadsheetml.template": 'xltx',
            "application/vnd.ms-excel.sheet.macroenabled.12": 'xlsm',
            "application/vnd.corelqpw": 'qpw',
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": 'pptx',
            "application/vnd.openxmlformats-officedocument.presentationml.slideshow": 'ppsx',
            "application/vnd.openxmlformats-officedocument.presentationml.slide": 'ppmx',
            "application/vnd.openxmlformats-officedocument.presentationml.template": 'potx',
            "application/vnd.ms-powerpoint": 'ppt',
            "application/vnd.ms-powerpoint.slideshow.macroenabled.12": 'ppsm',
            "application/vnd.ms-powerpoint.presentation.macroenabled.12": 'pptm',
            "application/vnd.ms-powerpoint.addin.macroenabled.12": 'ppam',
            "application/vnd.ms-powerpoint.slideshow": 'pps',
            "application/vnd.ms-powerpoint.presentation": 'ppt',
            "application/vnd.ms-powerpoint.addin": 'ppa',
            "message/rfc822": 'eml',
            "application/vnd.ms-outlook": 'msg',
            "application/mbox": 'mbox',
            "application/ost": 'ost',
            "application/emlx": 'emlx',
            "application/dbx": 'dbx',
            "application/dat": 'dat',
            "image/jpeg": 'jpg',
            "image/png": 'png',
            "image/gif": 'gif',
            "image/tiff": 'tiff',
            "image/bmp": 'bmp'
        }
        reverse_file_ext_map = {v: k for k, v in oFileExtMap.items()}

        if content_type_header not in oFileExtMap:
            log.error('Invalid file extension from content type header.')
            return 'Unsupported file format.', 400

        file_extension = oFileExtMap[content_type_header]
        if file_extension == 'use_extension':
            file_extension = os.path.splitext(filename)[1][1:].lower()

        log.info(
            f"Determined file extension: {file_extension} for file {filename}")

        pages = []
        num_pages = 0

        if file_extension == 'pdf':
            with open(temp_file_path, 'rb') as f:
                pdf_data = f.read()
            pages, num_pages = file_processor.split_pdf(pdf_data)
            del pdf_data

        elif file_extension in ['csv', 'xls', 'xltm', 'xltx', 'xlsx', 'tsv', 'ots']:
            with open(temp_file_path, 'rb') as f:
                excel_data = f.read()
            pages = file_processor.split_excel(excel_data)
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')
            del excel_data

        elif file_extension in ['eml', 'msg', 'pst', 'ost', 'mbox', 'dbx', 'dat', 'emlx']:
            with open(temp_file_path, 'rb') as f:
                email_data = f.read()
            pages = [("1", io.BytesIO(email_data))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')
            del email_data

        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp']:
            with open(temp_file_path, 'rb') as f:
                image_data = f.read()
            pages = [("1", io.BytesIO(image_data))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')
            del image_data

        elif file_extension == 'ods':
            with open(temp_file_path, 'rb') as f:
                ods_data = f.read()
            pages = file_processor.split_ods(ods_data)
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')
            del ods_data

        elif file_extension in [
            'docx', 'pdf', 'odt', 'odp', 'odg', 'odf', 'fodt', 'fodp', 'fodg',
            '123', 'dbf', 'scm', 'dotx', 'docm', 'dotm', 'xml', 'doc',
            'qpw', 'pptx', 'ppsx', 'ppmx', 'potx', 'pptm', 'ppam', 'ppsm',
            'pptm', 'ppam', 'ppt', 'pps', 'ppt', 'ppa', 'rtf'
        ]:
            pdf_data = convert_to_pdf(temp_file_path, file_extension)
            if pdf_data:
                pages, num_pages = file_processor.split_pdf(pdf_data)
            else:
                log.error('Conversion to PDF failed')
                return 'Conversion to PDF failed.', 400
        else:
            log.error('Unsupported file format')
            return 'Unsupported file format.', 400

        # Remove the original file from disk once we're done
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        bearer_token = google_auth.impersonated_id_token(
            serverurl=os.environ.get('SERVER_URL')).json()['token']
        log.info(f"bearer_token: {bearer_token}")
        headers['Content-Type'] = contentType
        headers['Authorization'] = f'Bearer {bearer_token}'

        loop = get_event_loop()
        results = loop.run_until_complete(process_pages_async(
            pages, headers, filename,
            namespace=project_id,
            file_id=file_id
        ))

        json_output = []
        processed_pages = 0
        for result, page_num in results:
            if not result.strip():
                log.info(
                    f"Skipping empty text in page {page_num} for JSON output.")
                continue
            page_obj = {
                'page': page_num,
                'text': result.strip()
            }
            json_output.append(page_obj)
            processed_pages += 1

        json_string = json.dumps(json_output, indent=4)
        log.info(f"Extraction successful for file: {filename}")

        # Suppose each processed page costs 1.5 credits
        message = json.dumps({
            "subscription_id": subscription_id,
            "user_id": user_id,
            "project_id": project_id,
            "creditsUsed": processed_pages * 1.5
        })
        log.info(f"Number of pages processed: {processed_pages}")
        log.info(f"Message to topic: {message}")
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
        )

        psql.update_file_status(
            id=file_id, status="processed",
            page_nums=processed_pages,
            updatedAt=datetime.now()
        )
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'}

    else:
        return 'Missing bucket or file name in event data', 400


def shutdown_handler(signal_int: int, frame) -> None:
    log.info(f"Caught Signal {signal.strsignal(signal_int)}")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
