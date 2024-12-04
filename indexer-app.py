import os
import io
from flask import Flask, request
from google.cloud import storage
from flask_limiter import Limiter, RequestLimit
from vertexai.preview.language_models import TextEmbeddingModel
from werkzeug.utils import secure_filename
from shared import file_processor, google_auth, security, google_pub_sub, psql
from types import FrameType
import json
import concurrent.futures
import multiprocessing
import time
from datetime import datetime
import sys
import tempfile
import requests
import aiohttp
import asyncio
import threading
import traceback
import ssl
import signal
from pinecone import Pinecone
sys.path.append('../')
from shared.logging_config import log
app = Flask(__name__)
app.debug=True

# Other environment-specific variables
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_CREDIT_USAGE_TOPIC = os.environ.get('GCP_CREDIT_USAGE_TOPIC')
UPLOADS_FOLDER = os.environ.get('UPLOADS_FOLDER')

app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
SERVER_URL = f"https://{os.environ.get('SERVER_URL')}/tika"

# Initialize connection to Pinecone
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX_NAME')

# Configure Pinecone client
pc = Pinecone(api_key=api_key)
# Connect to index
index = pc.Index(index_name)

def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Helper function to generate an MD5 hash from multiple arguments
def generate_md5_hash(*args):
    import json
    import hashlib
    # Serialize each argument with json.dumps
    serialized_args = [json.dumps(arg) for arg in args]
    # Join with '|'
    combined_string = '|'.join(serialized_args)
    # Compute MD5 hash
    md5_hash = hashlib.md5(combined_string.encode('utf-8')).hexdigest()
    return md5_hash

@app.route('/', methods=['POST'])
def event_handler():
    event_data = request.get_json()
    print(f"Event Data {event_data}")
    if not event_data:
        return 'No event data', 400

    try:
        bucket_name = event_data.get('bucket')  # Directly access 'bucket' property
        file_name = event_data.get('name')       # Directly access 'name' property
    except (KeyError, AttributeError):
        return 'Missing required fields in event data', 400

    if bucket_name and file_name:
        # Save the uploaded file to a temporary location
        folder_name, file_name_only = os.path.split(file_name)
        filename = file_name_only
        print(f"Filename {filename}")
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        content_type_header = download_file(bucket_name, file_name, temp_file_path)

        print(f'Folder: {folder_name}, File: {file_name_only}')

        # Extract subscription_id, project_id, user_id from folder_name
        folder_name_parts = folder_name.split('/')

        if len(folder_name_parts) >= 3:
            subscription_id = folder_name_parts[0]
            project_id = folder_name_parts[1]
            user_id = folder_name_parts[2]
        else:
            print('Invalid folder path format')
            return 'Invalid folder path format', 400

        print(f"Subscription ID: {subscription_id}, Project ID: {project_id}, User ID: {user_id}")

        # Generate file_id using MD5 hash
        file_id = generate_md5_hash(subscription_id, project_id, filename)
        print(f"Generated file_id: {file_id}")

        out_format_mapping = {
            'text': 'text/plain',
            'xml': 'application/xml',
            'html': 'text/html',
            'json': 'application/json',
            # Add more mappings as needed
        }

        # Define default values
        default_ocr_strategy = 'auto'
        default_out_format = 'text/plain'

        # Set values based on conditions
        x_tika_ocr_language = ''
        x_tika_pdf_ocr_strategy = default_ocr_strategy
        x_tika_accept = default_out_format

        # Now you can use these values in your headers
        headers = {
            'X-Tika-PDFOcrStrategy': x_tika_pdf_ocr_strategy,
            'Accept': x_tika_accept
        }

        # Add 'X-Tika-OCRLanguage' header only if value is not empty
        if x_tika_ocr_language:
            headers['X-Tika-OCRLanguage'] = x_tika_ocr_language

        print(f"\nContent-Type: {content_type_header}")

        if file_name_only:
            num_pages = 0
            # Update File status using file_id
            psql.update_file_status(id=file_id, status="processing", page_nums=num_pages, updatedAt=datetime.now())

            # Your existing file extension mapping code here
            oFileExtMap = {
                "application/octet-stream": "use_extension", # use file extension if content type is octet-stream
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
                "application/vnd.ms-word.document.macroenabled.12": 'docm',
                "application/vnd.ms-word.template.macroenabled.12": 'dotm',
                "application/rtf": 'rtf',
                "application/wordperfect": 'wpd',
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 'xlsx',
                "application/vnd.openxmlformats-officedocument.spreadsheetml.template": 'xltx',
                "application/vnd.ms-excel.sheet.macroenabled.12": 'xlsm',
                "application/vnd.ms-excel.template.macroenabled.12": 'xltm',
                "application/vnd.corelqpw": 'qpw',
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": 'pptx',
                "application/vnd.openxmlformats-officedocument.presentationml.slideshow": 'ppsx',
                "application/vnd.openxmlformats-officedocument.presentationml.slide": 'ppmx',
                "application/vnd.openxmlformats-officedocument.presentationml.template": 'potx',
                "application/vnd.ms-powerpoint": 'ppt',
                "application/vnd.ms-powerpoint.slideshow.macroenabled.12": 'ppsm',
                "application/vnd.ms-powerpoint.presentation.macroenabled.12": 'pptm',
                "application/vnd.ms-powerpoint.addin.macroenabled.12": 'ppam',
                "application/vnd.ms-powerpoint.slideshow.macroenabled.12": 'ppsm',
                "application/vnd.ms-powerpoint.presentation.macroenabled.12": 'pptm',
                "application/vnd.ms-powerpoint.addin.macroenabled.12": 'ppam',
                "application/vnd.ms-powerpoint": 'ppt',
                "application/vnd.ms-powerpoint.slideshow": 'pps',
                "application/vnd.ms-powerpoint.presentation": 'ppt',
                "application/vnd.ms-powerpoint.addin": 'ppa',
                # Email formats
                "message/rfc822": 'eml',  # EML format
                "application/vnd.ms-outlook": 'msg',  # MSG format
                "application/mbox": 'mbox',  # MBOX format
                "application/vnd.ms-outlook": 'pst',  # PST format
                "application/ost": 'ost',  # OST format
                "application/emlx": 'emlx',  # EMLX format
                "application/dbx": 'dbx',  # DBX format
                "application/dat": 'dat',  # Windows Mail (.dat) format
                # Image formats
                "image/jpeg": 'jpg',  # JPEG format
                "image/png": 'png',  # PNG format
                "image/gif": 'gif',  # GIF format
                "image/tiff": 'tiff',  # TIFF format
                "image/bmp": 'bmp'  # BMP format
            }
            # Reverse mapping of content types to file extensions
            reverse_file_ext_map = {v: k for k, v in oFileExtMap.items()}
            contentType = content_type_header
            if content_type_header not in oFileExtMap:
                print('Invalid file extension')
                return 'Unsupported file format.', 400
            
            file_extension = oFileExtMap[ content_type_header ]
            
            if file_extension == 'use_extension':
                file_extension = os.path.splitext(filename)[1][1:].lower()

            print(file_extension)
            print(filename)

            if file_extension == 'pdf':
                # Read the PDF file directly from memory
                # Read the file content as a buffer
                with open(temp_file_path, 'rb') as f:
                    pdf_data = f.read()

                # Measure the time taken to split the PDF
                start_time = time.time()
                pages, num_pages = file_processor.split_pdf(pdf_data)
                split_time = time.time() - start_time
                # log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
            
            elif file_extension in ['csv', 'xls', 'xltm', 'xltx', 'xlsx', 'tsv', 'ots']:
                with open(temp_file_path, 'rb') as f:
                    excel_data = f.read()
                pages = file_processor.split_excel(excel_data)
                num_pages = len(pages)
                contentType = reverse_file_ext_map.get(file_extension, '')

            elif file_extension in ['eml', 'msg', 'pst', 'ost', 'mbox', 'dbx', 'dat', 'emlx', ]:
                with open(temp_file_path, 'rb') as f:
                    email_data = f.read()
                pages = [("1", io.BytesIO(email_data))]
                num_pages = len(pages)
                contentType = reverse_file_ext_map.get(file_extension, '')

            elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp']:
                with open(temp_file_path, 'rb') as f:
                    image_data = f.read()
                pages = [("1", io.BytesIO(image_data))]
                num_pages = len(pages)
                contentType = reverse_file_ext_map.get(file_extension, '')

            elif file_extension in ['ods']:
                with open(temp_file_path, 'rb') as f:
                    ods_data = f.read()
                pages = file_processor.split_ods(ods_data)
                num_pages = len(pages)
                contentType = reverse_file_ext_map.get(file_extension, '')

            elif file_extension in ['docx', 'pdf', 'odt', 'odp', 'odg', 'odf', 'fodt', 'fodp', 'fodg', '123', 'dbf', 'html', 'scm', 'dotx', 'docm', 'dotm', 'xml', 'doc',  'qpw', 'pptx', 'ppsx', 'ppmx', 'potx', 'pptm', 'ppam', 'ppsm', 'pptm', 'ppam', 'ppt', 'pps', 'ppt', 'ppa', 'rtf']:
                # Convert the file to PDF using LibreOffice
                pdf_data = convert_to_pdf(temp_file_path, file_extension)

                if pdf_data:
                    # Measure the time taken to split the PDF
                    start_time = time.time()
                    pages, num_pages = file_processor.split_pdf(pdf_data)
                    split_time = time.time() - start_time
                    # log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
                    contentType = reverse_file_ext_map.get(file_extension, '')
                else:
                    log.error('Conversion to PDF failed')
                    return 'Conversion to PDF failed.', 400
            else:
                log.error('Unsupported file format')
                return 'Unsupported file format.', 400

            # Create the ID token
            bearer_token = google_auth.impersonated_id_token(serverurl=os.environ.get('SERVER_URL')).json()['token']
            log.info(f"bearer_token: {bearer_token}")
            # Append Header value
            headers['Content-Type'] = contentType
            headers['Authorization'] = f'Bearer {bearer_token}'

            loop = get_event_loop()
            results = loop.run_until_complete(process_pages_async(pages, headers, filename, namespace=project_id, file_id=file_id))

            # Build the JSON output using mapped_results
            json_output = []
            for result, page_num in results:
                page_obj = {
                    'page': page_num,
                    'text': result.strip()
                }
                json_output.append(page_obj)

            json_string = json.dumps(json_output, indent=4)
            log.info(f"Extraction successful for file: {filename}")

            # Send CREDIT USAGE TO TOPIC
            message = json.dumps({
                "subscription_id": subscription_id,
                "user_id": user_id,
                "project_id": project_id,
                "creditsUsed": num_pages
            })
            log.info(f"Number of pages processed: {num_pages}")
            log.info(f"Message to topic: {message}")
            google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message)

            # Update File status in File table
            sqlres = psql.update_file_status(id=file_id, status="processed", page_nums=num_pages, updatedAt=datetime.now())
            return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'}

        else:
            log.error('No file uploaded')
            return 'No file uploaded.', 400

    else:
        return 'Missing bucket or file name in event data', 400

async def get_google_embedding(queries):
    embedder_name = "text-multilingual-embedding-preview-0409"
    model = TextEmbeddingModel.from_pretrained(embedder_name)
    embeddings_list = model.get_embeddings(queries)
    embeddings = [embedding.values for embedding in embeddings_list]
    return embeddings

def download_file(bucket_name, filename, temp_file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(filename)
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

    blob.download_to_filename(temp_file_path)
    print(f'Downloaded {filename} from bucket {bucket_name} to {temp_file_path}')
    return blob.content_type

def convert_to_pdf(file_path, file_extension):
    import subprocess

    try:
        # Specify the output PDF file path
        pdf_file_path = os.path.splitext(file_path)[0] + '.pdf'

        # Convert the file to PDF using LibreOffice
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

            # Delete the output PDF file
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

async def process_pages_async(pages, headers, filename, namespace, file_id):
    url = SERVER_URL
    async with aiohttp.ClientSession() as session:
        tasks = [async_put_request(session, url, page_data, page_num, headers) for page_num, page_data in pages]
        results = await asyncio.gather(*tasks)

    # Collect embeddings and upload in batches
    await create_and_upload_embeddings_in_batches(results, filename, namespace, file_id)
    return results

async def async_put_request(session, url, payload, page_num, headers, max_retries=10):
    retries = 0

    while retries < max_retries:
        try:
            payload_copy = io.BytesIO(payload.getvalue())
            async with session.put(url, data=payload_copy, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 429:
                    print(f"Retrying request for page {page_num}, Retry #{retries + 1}")
                    retries += 1
                    await asyncio.sleep(1)  # You may adjust the sleep duration
                    continue  # Retry the request
                content = await response.read()  # Read the content of the response
                text_content = content.decode('utf-8', errors='ignore')  # Decode bytes to Unicode text, ignoring errors
                return text_content, page_num

        except aiohttp.ClientError as e:
            print(f"Error during request for page {page_num}: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except ssl.SSLError as e:
            print(f"SSL Error during request for page {page_num}: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except asyncio.TimeoutError:
            print(f"Timeout error during request for page {page_num}. Retrying...")
            retries += 1
            await asyncio.sleep(1)  # Adjust the sleep duration

    # If retries are exhausted, raise an exception or handle it as needed
    raise RuntimeError(f"Failed after {max_retries} retries for page {page_num}")

async def create_and_upload_embeddings_in_batches(results, filename, namespace, file_id, batch_size=10):
    batch = []
    texts = []
    page_nums = []

    for text_content, page_num in results:
        texts.append(text_content)
        page_nums.append(page_num)

        if len(texts) >= batch_size:
            embeddings = await get_google_embedding(texts)
            for text, embedding, page_num in zip(texts, embeddings, page_nums):
                # Use file_id as prefix in document_id
                document_id = f"{file_id}#{page_num}"
                metadata = {
                    "text": text,
                    "source": filename,
                    "page": page_num
                }
                batch.append({
                    "id": document_id,
                    "values": embedding,
                    "metadata": metadata
                })
            upload_to_pinecone(batch, namespace)
            batch.clear()
            texts.clear()
            page_nums.clear()

    # Process any remaining items
    if texts:
        embeddings = await get_google_embedding(texts)
        for text, embedding, page_num in zip(texts, embeddings, page_nums):
            # Use file_id as prefix in document_id
            document_id = f"{file_id}#{page_num}"
            metadata = {
                "text": text,
                "source": filename,
                "page": page_num
            }
            batch.append({
                "id": document_id,
                "values": embedding,
                "metadata": metadata
            })
        upload_to_pinecone(batch, namespace)

def upload_to_pinecone(vectors, namespace):
    indexres = index.upsert(vectors=vectors, namespace=namespace)
    print(f"Upsert response {indexres}")
    return indexres

def shutdown_handler(signal_int: int, frame) -> None:
    log.info(f"Caught Signal {signal.strsignal(signal_int)}")

    # Safely exit program
    sys.exit(0)

if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # Handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    # Handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
