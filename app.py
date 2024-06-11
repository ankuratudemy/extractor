import os
import io
from flask import Flask, request, make_response, render_template, jsonify, abort, make_response, Response
from flask_limiter import Limiter, RequestLimit
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from vertexai.preview.language_models import TextEmbeddingModel
import json
from shared import file_processor, google_auth, security, google_pub_sub
from types import FrameType
import json
import concurrent.futures
import multiprocessing
import time
import sys
import tempfile
import requests
import aiohttp
import asyncio
import threading
import ssl
import signal

sys.path.append('../')
from shared.logging_config import log
app = Flask(__name__)
app.debug=True

# Assuming you have Redis connection details
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
SECRET_KEY = os.environ.get('SECRET_KEY')
PINECONE_API_KEY= os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME= os.environ.get('PINECONE_INDEX_NAME')
#Other env specific varibales:

GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCP_CREDIT_USAGE_TOPIC = os.environ.get('GCP_CREDIT_USAGE_TOPIC')
UPLOADS_FOLDER = os.environ.get('UPLOADS_FOLDER')


def verify_api_key():
    api_key_header = request.headers.get('API-KEY')
    res = security.api_key_required(api_key_header)
    return res


@app.before_request
def before_request():
    log.info(f"request.endpoint {request.endpoint}")
    if request.endpoint == 'extract' or request.endpoint == 'search':  # Check if the request is for the /extract route
        valid = verify_api_key()
        if not valid:
            return Response(status=401)
        
def default_error_responder(request_limit: RequestLimit):
    # return jsonify({"message": f'rate Limit Exceeded: {request_limit}'}), 429
    return Response(status=429)

limiter = Limiter(
        key_func=lambda: getattr(request, 'tenant_data', {}).get('tenant_id', None),
        app=app,
        storage_uri=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/xtract",
        storage_options={"socket_connect_timeout": 30},
        strategy="moving-window",  # or "moving-window"
        on_breach=default_error_responder
    )

app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
SERVER_URL = f"https://{os.environ.get('SERVER_URL')}/tika"


# Create the ID token
bearer_token = google_auth.impersonated_id_token(serverurl=os.environ.get('SERVER_URL')).json()['token']

def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

@app.route('/health', methods=['GET'], endpoint='health_check')
def health_check():
    return json.dumps({"status": "ok"})

@app.route('/extract', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def extract():
    lang_value = request.form.get('lang', '')
    ocr_value = request.form.get('ocr', '')
    out = request.form.get('out_format', '')
    
    # # Define mapping for lang_value
    # lang_mapping = {
    #     'hindi': 'hin',
    #     # Add more mappings as needed
    # }

    # Define mapping for lang_value
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
    x_tika_ocr_language = lang_value
    x_tika_pdf_ocr_strategy = 'no_ocr' if ocr_value.lower() == 'false' else ('ocr_only' if ocr_value.lower() == 'true' else default_ocr_strategy)
    x_tika_accept = out_format_mapping.get(out, default_out_format)


    # Now you can use these values in your headers
    headers = {
        'X-Tika-PDFOcrStrategy': x_tika_pdf_ocr_strategy,
        'Accept': x_tika_accept
    }

    # Add 'X-Tika-OCRLanguage' header only if value is not empty
    if x_tika_ocr_language:
        headers['X-Tika-OCRLanguage'] = x_tika_ocr_language

    # Access a specific header
    content_type_header = request.headers.get('Content-Type')
    log.info(f"\nContent-Type Header: {content_type_header}")

    contentType='application/pdf'
    uploaded_file = request.files['file']

    if uploaded_file:
        num_pages=0
        # Save the uploaded file to a temporary location
        filename = secure_filename(uploaded_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        log.info(uploaded_file.content_type)
        
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

        if uploaded_file.content_type not in oFileExtMap:
            log.info('Invalid file extension')
            return 'Unsupported file format.', 400
        
        file_extension = oFileExtMap[ uploaded_file.content_type ]
        
        if file_extension == 'use_extension':
            file_extension = os.path.splitext(filename)[1][1:].lower()

        print(file_extension)
        print(filename)

        if file_extension == 'pdf':
            # Read the PDF file directly from memory
            pdf_data = uploaded_file.read()

            # Measure the time taken to split the PDF
            start_time = time.time()
            pages, num_pages = file_processor.split_pdf(pdf_data)
            split_time = time.time() - start_time
            # log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
        
        elif file_extension in ['csv', 'xls', 'xltm', 'xltx', 'xlsx', 'tsv', 'ots']:
            pages = file_processor.split_excel(uploaded_file.read())
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['eml', 'msg', 'pst', 'ost', 'mbox', 'dbx', 'dat', 'emlx', ]:
            pages = [("1", io.BytesIO(uploaded_file.read()))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp']:
            pages = [("1", io.BytesIO(uploaded_file.read()))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['ods']:
            pages = file_processor.split_ods(uploaded_file.read())
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['docx', 'pdf', 'odt', 'odp', 'odg', 'odf', 'fodt', 'fodp', 'fodg', '123', 'dbf', 'html', 'scm', 'dotx', 'docm', 'dotm', 'xml', 'doc',  'qpw', 'pptx', 'ppsx', 'ppmx', 'potx', 'pptm', 'ppam', 'ppsm', 'pptm', 'ppam', 'ppt', 'pps', 'ppt', 'ppa', 'rtf']:
            uploaded_file.save(temp_file_path)
            # Convert the file to PDF using LibreOffice
            pdf_data = convert_to_pdf(temp_file_path, file_extension)

            if pdf_data:
                # Measure the time taken to split the PDF
                start_time = time.time()
                pages, num_pages = file_processor.split_pdf(pdf_data)
                split_time = time.time() - start_time
                # log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
            else:
                log.error('Conversion to PDF failed')
                return 'Conversion to PDF failed.', 400
        else:
            log.error('Unsupported file format')
            return 'Unsupported file format.', 400

        # # Get the number of available CPUs
        # num_cpus = multiprocessing.cpu_count()
        # log.info(num_cpus)

        #Append Header value
        headers['Content-Type'] = contentType
        headers['Authorization'] = f'Bearer {bearer_token}'

        loop = get_event_loop()
        results = loop.run_until_complete(process_pages_async(pages, headers))

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

        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        #send CREDIT USAGE TO TOPIC 'structhub-credit-usage'

        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
        "username": getattr(request, 'tenant_data', {}).get('tenant_id', None),
        "creditsUsed": num_pages
        })
        log.info(f"Number of pages processed: {num_pages}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=message)
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'}

    else:
        log.error('No file uploaded')
        return 'No file uploaded.', 400

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
            # log.error('Conversion to PDF failed: Output file not found')
            return None
    except subprocess.CalledProcessError as e:
        log.error(f'Conversion to PDF failed: {str(e)}')
        return None
    except Exception as e:
        log.error(f'Error during PDF conversion: {str(e)}')
        return None

async def process_pages_async(pages, headers):
    url = SERVER_URL
    async with aiohttp.ClientSession() as session:
        tasks = [async_put_request(session, url, page_data, page_num, headers) for page_num, page_data in pages]
        results = await asyncio.gather(*tasks)

    # Map results to their corresponding page numbers
    mapped_results = [(result, page_num) for (result, page_num) in results]

    return mapped_results

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

@app.route('/search', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def search():
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        log.info(f"user_id {userId}")
        if not userId:
            return 'invalid request', 400
    
        # Check if request body contains the required parameters
        if not request.is_json:
            return 'invalid request', 400
        
        data = request.get_json()
        log.info(data)
        if 'topk' not in data or 'query' not in data:
            missing_params = []
            if 'topk' not in data:
                missing_params.append('topk')
            if 'query' not in data:
                missing_params.append('query')
            return f"Missing required parameters: {', '.join(missing_params)}", 400
        
        topk = data['topk']
        query = data['query']
        log.info(f"query: {query} topK: {topk}")
    except Exception as e:
        log.info(str(e))
        return 'validation: Something went wrong', 500
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        log.info(str(e))
        return 'vectorDB: Something went wrong', 500

    try:
        #Get google embeddings:
        embeddings = asyncio.run(get_google_embedding(queries=[query]))
    except Exception as e:
        log.error(str(e))
        return 'vectorDB Search: Something went wrong', 500

    try:
        # Search Vector DB
        docs = index.query(
            namespace=str(userId),
            vector=embeddings[0],
            top_k=topk,
            include_values=False,
            include_metadata=True,
            # filter={
            #     "genre": {"$in": ["comedy", "documentary", "drama"]}
            # }
        )
        log.info(f"Docs: {docs.to_dict()}")
        doc_list = [{'source': match['metadata']['source'], 'page': match['metadata']['page'], 'text': match['metadata']['text']} for match in docs.to_dict()['matches']]
        count = len(doc_list)
        final_response = {
            'count': count,
            'data': doc_list
        }
        json_string = json.dumps(final_response, indent=4)
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'} 
    except Exception as e:
        log.error(str(e))
        return 'Query index: Something went wrong', 500

async def get_google_embedding(queries):
    embedder_name = "text-multilingual-embedding-preview-0409"
    model = TextEmbeddingModel.from_pretrained(embedder_name)
    embeddings_list = model.get_embeddings(queries)
    embeddings = [embedding.values for embedding in embeddings_list]
    return embeddings
def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    log.info(f"Caught Signal {signal.strsignal(signal_int)}")

    # Safely exit program
    sys.exit(0)
if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)