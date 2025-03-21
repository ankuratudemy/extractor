import os
import io
import re
from typing import List
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from google.genai import types
from flask import Flask, request, make_response, render_template, jsonify, abort, make_response, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter, RequestLimit
from werkzeug.utils import secure_filename
from langchain.chains import qa_with_sources
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone as PC
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
# BM25 + sparse vector utilities (with no spaCy usage)
from shared.bm25 import (
    tokenize_document,
    compute_bm25_sparse_vector,
    get_project_vocab_stats
)
from langchain_core.outputs import LLMResult
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
callbacks = [FinalStreamingStdOutCallbackHandler()]
from vertexai.preview.language_models import TextEmbeddingModel
from urllib.parse import urlencode
import json
from shared import file_processor, google_auth, security, google_pub_sub, search_helper
from types import FrameType
import json
import time
import sys
import aiohttp
import asyncio
import ssl
import signal
import tiktoken

sys.path.append('../')
from shared.logging_config import log
app = Flask(__name__)
CORS(app, origins="*")
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

anthropic_chat = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet@20240620",
    project=GCP_PROJECT_ID,
    location="us-east5",
    max_tokens=4096,
    temperature=0.1
)

chat_stream = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    streaming=True
    # api_key="" # Optional if not set as an environment variable
)
chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}}
    # api_key="" # Optional if not set as an environment variable
)
vertexchat = ChatVertexAI(model="gemini-1.5-pro", safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    })
vertexchat_stream = ChatVertexAI(model="gemini-2.0-flash-exp", response_mime_type="application/json", safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    })

google_genai_client = genai.Client(
    vertexai=True, project=GCP_PROJECT_ID, location='us-central1'
)
@app.before_request
def before_request():
    log.info(f"Request endpoint: {request.endpoint}")
    protected_endpoints = [
        'extract', 'search', 'chat', 'groqchat', 'geminichat',
        'anthropic', 'serp', 'webextract', 'qna', 'chatmr', 'askme'
    ]
    if request.endpoint in protected_endpoints and request.method != 'OPTIONS':
        api_key_header = request.headers.get('API-KEY')
        valid = security.api_key_required(api_key_header)
        def generate():
            yield 'No credits left to process request!'
        if not valid and (request.endpoint == 'anthropic' or request.endpoint == 'askme') :
            return Response(stream_with_context(generate()), mimetype='text/event-stream') 
        if not valid:
            return Response(status=401)
        
def default_error_responder(request_limit: RequestLimit):
    # return jsonify({"message": f'rate Limit Exceeded: {request_limit}'}), 429
    return Response(status=429)

limiter = Limiter(
        key_func=lambda: getattr(request, 'tenant_data', {}).get('subscription_id', None),
        app=app,
        storage_uri=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/xtract",
        storage_options={"socket_connect_timeout": 30},
        strategy="moving-window",  # or "moving-window"
        on_breach=default_error_responder
    )

app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER
SERVER_URL = f"https://{os.environ.get('SERVER_URL')}/tika"
WEBSEARCH_SERVER_URL = f"https://{os.environ.get('WEBSEARCH_SERVER_URL')}/search"


# Create the ID token
bearer_token = google_auth.impersonated_id_token(serverurl=os.environ.get('SERVER_URL')).json()['token']
websearch_bearer_token = google_auth.impersonated_id_token(serverurl=os.environ.get('WEBSEARCH_SERVER_URL')).json()['token']

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
        
        # Reverse mapping of content types to file extensions
        reverse_file_ext_map = {v: k for k, v in oFileExtMap.items()}

        if uploaded_file.content_type not in oFileExtMap:
            log.info('Invalid file extension')
            return 'Unsupported file format.', 400
        
        file_extension = oFileExtMap[ uploaded_file.content_type ]
        
        if file_extension == 'use_extension':
            file_extension = os.path.splitext(filename)[1][1:].lower()

        log.info(file_extension)
        log.info(filename)

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

        elif file_extension in ['jpg', 'jpeg', 'png', 'gif', 'tiff', 'bmp', 'html']:
            pages = [("1", io.BytesIO(uploaded_file.read()))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['ods']:
            pages = file_processor.split_ods(uploaded_file.read())
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['docx', 'pdf', 'odt', 'odp', 'odg', 'odf', 'fodt', 'fodp', 'fodg', '123', 'dbf', 'scm', 'dotx', 'docm', 'dotm', 'xml', 'doc',  'qpw', 'pptx', 'ppsx', 'ppmx', 'potx', 'pptm', 'ppam', 'ppsm', 'pptm', 'ppam', 'ppt', 'pps', 'ppt', 'ppa', 'rtf']:
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
        chargeable_credits = 0
        for result, page_num in results:
            if result.strip():  # Check if the text is not empty after stripping
                chargeable_credits += 1
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
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": chargeable_credits
        })
        log.info(f"Number of pages processed: {num_pages}")
        log.info(f" Chargeable creadits: {chargeable_credits}")
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
                    log.info(f"Retrying request for page {page_num}, Retry #{retries + 1}")
                    retries += 1
                    await asyncio.sleep(1)  # You may adjust the sleep duration
                    continue  # Retry the request
                content = await response.read()  # Read the content of the response
                text_content = content.decode('utf-8', errors='ignore')  # Decode bytes to Unicode text, ignoring errors
                return text_content, page_num

        except aiohttp.ClientError as e:
            log.info(f"Error during request for page {page_num}: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except ssl.SSLError as e:
            log.info(f"SSL Error during request for page {page_num}: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except asyncio.TimeoutError:
            log.info(f"Timeout error during request for page {page_num}. Retrying...")
            retries += 1
            await asyncio.sleep(1)  # Adjust the sleep duration

    # If retries are exhausted, raise an exception or handle it as needed
    raise RuntimeError(f"Failed after {max_retries} retries for page {page_num}")


@app.route('/search', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def search():
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        projectId = getattr(request, 'tenant_data', {}).get('project_id', None)
        namespace = projectId
        log.info(f"user_id {userId}")
        log.info(f"project id to search(namespace) {projectId}")
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
        log.info(f"query: {query} topk: {topk}")
    except Exception as e:
        log.info(str(e))
        return 'validation: Something went wrong', 500
    
    try:
        pc = PC(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
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
            namespace=str(namespace),
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
        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": 0.2
        })
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=message)
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'} 
    except Exception as e:
        log.error(str(e))
        return 'Query index: Something went wrong', 500

async def async_get_request(session, url, params, headers=None, max_retries=10):
    retries = 0
    query_string = urlencode(params)

    while retries < max_retries:
        try:
            async with session.get(f"{url}?{query_string}", headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
                log.info(f"Request URL: {url}?{query_string}")
                log.info(f"Response status: {response.status}")
                log.info(f"Response headers: {response.headers}")
                
                if response.status == 429:
                    log.info(f"Retrying websearch request, Retry #{retries + 1}")
                    retries += 1
                    await asyncio.sleep(1)  # You may adjust the sleep duration
                    continue  # Retry the request
                
                content = await response.read()  # Read the content of the response
                log.info(f"Response content: {content}")
                
                try:
                    json_content = json.loads(content)
                except json.JSONDecodeError:
                    log.error(f"Failed to decode JSON response: {content}")
                    raise RuntimeError("Invalid JSON response")
                
                return json_content, params.get('pageno', 1)

        except aiohttp.ClientError as e:
            log.info(f"Error during websearch: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except ssl.SSLError as e:
            log.info(f"SSL Error during websearch: {str(e)}")
            retries += 1
            await asyncio.sleep(1)  # You may adjust the sleep duration

        except asyncio.TimeoutError:
            log.info(f"Timeout error during websearch. Retrying...")
            retries += 1
            await asyncio.sleep(1)  # Adjust the sleep duration

    # If retries are exhausted, raise an exception or handle it as needed
    raise RuntimeError(f"Failed after {max_retries} retries for websearch")

@app.route('/serp', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def serp():
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        log.info(f"user_id {userId}")
        if not userId:
            return 'invalid request', 400
        
        if not request.is_json:
            return 'invalid request', 400
        
        data = request.get_json()
        log.info(data)
        headers = {}
        headers['Authorization'] = f'Bearer {websearch_bearer_token}'
        required_params = ['q', 'format', 'time_range', 'count']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}", 400

        query = data['q']
        count = data['count']
        params = {param: data[param] for param in required_params}
        params['pageno'] = 1  # Initialize page number
        log.info(f"params: {params}")

    except Exception as e:
        log.error(str(e))
        return 'validation: Something went wrong', 500
    
    try:
        total_results = []
        total_count = 0

        async def fetch_results():
            async with aiohttp.ClientSession() as session:
                nonlocal total_results, total_count
                while total_count < count:
                    result, pageno = await async_get_request(session, WEBSEARCH_SERVER_URL, params, headers)
                    results = result.get('results', [])
                    if not results:
                        break
                    total_results.extend(results)
                    total_count = len(total_results)
                    log.info(f"Total results fetched: {total_count}")
                    params['pageno'] = pageno + 1  # Increment page number

        asyncio.run(fetch_results())
        final_results = total_results[:count]
        json_string = json.dumps(final_results, indent=4)
        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": count * 0.2
        })
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=message)
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'}
    except Exception as e:
        log.error(str(e))
        return 'Websearch call: Something went wrong', 500

@app.route('/webextract', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def webextract():
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        log.info(f"user_id {userId}")
        if not userId:
            return 'invalid request', 400
        
        if not request.is_json:
            return 'invalid request', 400
        
        data = request.get_json()
        log.info(data)
        headers = {}
        headers['Authorization'] = f'Bearer {websearch_bearer_token}'
        required_params = ['q', 'time_range', 'count']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}", 400
        query = data['q']
        count = data['count']
        params = {param: data[param] for param in required_params}
        params['pageno'] = 1  # Initialize page number
        params['format'] = 'json' # set output format as json
        log.info(f"params: {params}")

    except Exception as e:
        log.error(str(e))
        return 'validation: Something went wrong', 500
    
    try:
        total_results = []
        total_count = 0
        processed_urls = set()

        async def fetch_results():
            async with aiohttp.ClientSession() as session:
                nonlocal total_results, total_count
                while total_count < count:
                    result, pageno = await async_get_request(session, WEBSEARCH_SERVER_URL, params, headers)
                    results = result.get('results', [])
                    if not results:
                        break
                    for item in results:
                        if item['url'] not in processed_urls:
                            processed_urls.add(item['url'])
                            total_results.append(item)
                            total_count += 1
                            if total_count >= count:
                                break
                    params['pageno'] = pageno + 1  # Increment page number

        asyncio.run(fetch_results())
        final_results = total_results[:count]

        # Fetch HTML content and extract text for each URL
        async def fetch_and_extract(url, title, headers):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html_content = await response.read()
                    payload = io.BytesIO(html_content)
                    extract_headers = {
                        'Content-Type': 'text/html',
                        'Accept': 'text/plain',
                        'Authorization': f'Bearer {bearer_token}'
                    }
                    text, _ = await async_put_request(session, SERVER_URL, payload, title, extract_headers)
                    return text

        loop = get_event_loop()
        tasks = [fetch_and_extract(item['url'], item['title'], headers) for item in final_results]
        extracted_texts = loop.run_until_complete(asyncio.gather(*tasks))

        json_output = []
        for text, item in zip(extracted_texts, final_results):
            page_obj = {
                'url': item['url'],
                'title': item['title'],
                'text': text.strip()
            }
            json_output.append(page_obj)

        json_string = json.dumps(json_output, indent=4)
        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": count * 0.3
        })
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=message)
        return json_string, 200, {'Content-Type': 'application/json; charset=utf-8'}

    except Exception as e:
        log.error(str(e))
        return 'Websearch call: Something went wrong', 500
    
async def getVectorStoreDocs(request):
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        projectId = getattr(request, 'tenant_data', {}).get('project_id', None)
        namespace = projectId
        log.info(f"user_id {userId}")
        log.info(f"project id to search(namespace) {projectId}")

        if not userId:
            raise Exception("invalid request")

        if not request.is_json:
            raise Exception("invalid request")

        data = request.get_json()
        log.info(data)
        if 'topk' not in data or 'q' not in data:
            missing_params = []
            if 'topk' not in data:
                missing_params.append('topk')
            if 'q' not in data:
                missing_params.append('q')
            return f"Missing required parameters: {', '.join(missing_params)}", 400

        topk = data['topk']
        query = data['q']
        history = data.get('history', [])
        log.info(f"query: {query} topk: {topk}")

    except Exception as e:
        log.info(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # Setup Pinecone Index
    try:
        pc = PC(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
    except Exception as e:
        log.info(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # Generate multiple queries
    try:
        multi_query_prompt = search_helper.create_multi_query_prompt(history, query)
        # multi_query_resp = anthropic_chat.invoke(input=multi_query_prompt).content
        sync_response = google_genai_client.models.generate_content(
                                 model='gemini-2.0-flash-exp',
                                 contents=multi_query_prompt,
                                 config= {
                                 'temperature': 0.5
                                 }
                            )
        multi_query_resp = sync_response.text
        all_queries = [q.strip() for q in multi_query_resp.split('\n') if q.strip()]
        log.info(f"Generated Queries: {all_queries}")

        keyword_query = all_queries[-1]
        try:
            keywords = json.loads(keyword_query)
            if not isinstance(keywords, list):
                keywords = [keyword_query]
        except:
            keywords = [keyword_query]

        search_queries = all_queries[:-1] + keywords

    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # Get embeddings for all queries
    try:
        embeddings_list = await get_google_embedding(queries=search_queries)
        for query in search_queries:
            tokens = tokenize_document(combined_text)
            # 3) Publish partial update => aggregator merges term freq
            publish_partial_bm25_update(project_id, tokens, is_new_doc=True)

            # 4) Get BM25 stats from Firestore => { "N":..., "avgdl":..., "vocab": { term-> {df, tf} } }
            vocab_stats = get_project_vocab_stats(project_id)

            return compute_bm25_sparse_vector(tokens, project_id, vocab_stats, max_terms=max_terms)
    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # Perform Pinecone searches for each query
    all_matches = []
    try:
        for sq, emb in zip(search_queries, embeddings_list):
            docs = index.query(
                namespace=str(namespace),
                vector=emb,
                top_k=topk,
                include_values=False,
                include_metadata=True
            ).to_dict().get('matches', [])
            all_matches.extend(docs)

        # Deduplicate based on doc id
        unique_docs = {}
        for doc in all_matches:
            doc_id = doc['id']
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc

        final_docs = list(unique_docs.values())
        # Convert final_docs to required format
        doc_list = []
        for match in final_docs:
            doc_list.append({
                'source': match['metadata']['source'],
                'page': match['metadata']['page'],
                'text': match['metadata']['text']
            })

        # Reorder docs based on regex keyword matches
        doc_list = await search_helper.reorder_docs(doc_list, keywords)

        # Now pick up to topk docs without exceeding 50k tokens
        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = 50000

        topk_docs = []
        total_tokens = 0
        for i, doc in enumerate(doc_list):
            if len(topk_docs) >= topk:
                # Reached topk limit
                break

            # Encode this doc alone to count tokens
            doc_str = json.dumps(doc)
            doc_tokens = len(enc.encode(doc_str))

            # Check if adding this doc exceeds token limit
            if total_tokens + doc_tokens > max_tokens:
                # Adding this doc would exceed token limit, stop adding more
                break

            # Add the doc
            topk_docs.append(doc)
            total_tokens += doc_tokens

        log.info(f"Final topk_docs count: {len(topk_docs)}, total_tokens: {total_tokens}")

        # Prepare usage message
        count = len(topk_docs)
        message = json.dumps({
            "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
            "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
            "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
            "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
            "creditsUsed": count * 0.2
        })
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message)

        # Respond with the final topk_docs
        return json.dumps(topk_docs), 200

    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

async def getWebExtract(request):
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        log.info(f"user_id {userId}")
        if not userId:
            raise Exception("invalid request")

        if not request.is_json:
            raise Exception("invalid request")
   
        data = request.get_json()
        log.info(data)
        headers = {}
        headers['Authorization'] = f'Bearer {websearch_bearer_token}'
        required_params = ['q', 'time_range', 'count']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}", 400
        query = data['q']
        count = data['count']
        params = {param: data[param] for param in required_params}
        params['pageno'] = 1  # Initialize page number
        params['format'] = 'json'  # Set output format as json
        log.info(f"params: {params}")

    except Exception as e:
        log.error(str(e))
        raise Exception(str(e))

    try:
        total_results = []
        total_count = 0
        processed_urls = set()
        target_count = count * 3  # Fetch 3 times the number of URLs

        async def fetch_results():
            async with aiohttp.ClientSession() as session:
                nonlocal total_results, total_count
                while total_count < target_count:
                    result, pageno = await async_get_request(session, WEBSEARCH_SERVER_URL, params, headers)
                    results = result.get('results', [])
                    if not results:
                        break
                    for item in results:
                        if item['url'] not in processed_urls:
                            processed_urls.add(item['url'])
                            total_results.append(item)
                            total_count += 1
                            if total_count >= target_count:
                                break
                    params['pageno'] = pageno + 1  # Increment page number

        await fetch_results()
        final_results = total_results[:target_count]

        async def fetch_and_extract(url, title, headers):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        log.info(response.status)
                        log.info(response)
                        if response.status != 200:
                            return ''  # Skip if non-200 response

                        content_type = response.headers.get('Content-Type', '').lower()
                        log.info(f"content type: {content_type}")
                        content = await response.read()
                        file_extension = oFileExtMap.get(content_type, None)
                        log.info(f"file extension: {file_extension}")
                        if file_extension:
                            payload = io.BytesIO(content)
                            extract_headers = {
                                'Accept': 'text/plain',
                                'Authorization': f'Bearer {bearer_token}'
                            }
                            text, _ = await async_put_request(session, SERVER_URL, payload, title, extract_headers)
                            log.info(f"TEXT EXTRACTED: {text}")
                            return text
                        else:
                            return ''  # Unsupported content type
            except Exception as e:
                log.error(f"Exception occurred while processing URL {url}: {e}")
                return ''  # Continue processing other URLs even if an exception occurs

        tasks = [fetch_and_extract(item['url'], item['title'], headers) for item in final_results]
        extracted_texts = await asyncio.gather(*tasks)

        # Filter out empty texts and limit to the desired count
        non_empty_texts = [text for text in extracted_texts if text.strip()]
        final_texts = non_empty_texts[:count]

        json_output = []
        for text, item in zip(final_texts, final_results[:len(final_texts)]):
            page_obj = {
                'source': item['url'],
                'page': item['title'],
                'text': text.strip()
            }
            json_output.append(page_obj)

        json_string = json.dumps(json_output, indent=4)
        # Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
            "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
            "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
            "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
            "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
            "creditsUsed": count * 0.3
        })
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message)
        return json_string

    except Exception as e:
        log.error(str(e))
        raise Exception(str(e))
    

async def getWebExtractLCFormat(request):
    try:
        userId = getattr(request, 'tenant_data', {}).get('user_id', None)
        log.info(f"user_id {userId}")
        if not userId:
            raise Exception("invalid request")
        
        if not request.is_json:
            raise Exception("invalid request")
        
        data = request.get_json()
        log.info(data)
        headers = {}
        headers['Authorization'] = f'Bearer {websearch_bearer_token}'
        required_params = ['q', 'time_range', 'count']
        missing_params = [param for param in required_params if param not in data]
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}", 400
        query = data['q']
        count = data['count']
        params = {param: data[param] for param in required_params}
        params['pageno'] = 1  # Initialize page number
        params['format'] = 'json' # set output format as json
        log.info(f"params: {params}")

    except Exception as e:
        log.error(str(e))
        raise Exception(str(e))
    
    try:
        total_results = []
        total_count = 0
        processed_urls = set()

        async def fetch_results():
            async with aiohttp.ClientSession() as session:
                nonlocal total_results, total_count
                while total_count < count:
                    result, pageno = await async_get_request(session, WEBSEARCH_SERVER_URL, params, headers)
                    results = result.get('results', [])
                    if not results:
                        break
                    for item in results:
                        if item['url'] not in processed_urls:
                            processed_urls.add(item['url'])
                            total_results.append(item)
                            total_count += 1
                            if total_count >= count:
                                break
                    params['pageno'] = pageno + 1  # Increment page number

        await fetch_results()
        final_results = total_results[:count]

        # Fetch HTML content and extract text for each URL
        async def fetch_and_extract(url, title, headers):
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    html_content = await response.read()
                    payload = io.BytesIO(html_content)
                    extract_headers = {
                        'Content-Type': 'text/html',
                        'Accept': 'text/plain',
                        'Authorization': f'Bearer {bearer_token}'
                    }
                    text, _ = await async_put_request(session, SERVER_URL, payload, title, extract_headers)
                    # Replace consecutive newlines and tabs
                    text = re.sub(r'\n+', '\n', text)
                    text = re.sub(r'\t+', '\t', text)
                    text = re.sub(r'\\n', '', text)
                    return text

        loop = asyncio.get_event_loop()
        tasks = [fetch_and_extract(item['url'], item['title'], headers) for item in final_results]
        extracted_texts = await asyncio.gather(*tasks)

        document_out = []
        for text, item in zip(extracted_texts, final_results):
            page_obj = Document(
                page_content=text.strip(),
                metadata={'source': item['url'],
                'page': item['title']}
            )
            document_out.append(page_obj)

        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps({
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": count
        })
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=message)
        return document_out

    except Exception as e:
        log.error(str(e))
        raise Exception(str(e))


def split_prompt(prompt, max_tokens=20000):
    """Splits the prompt into chunks that fit within the max token limit."""
    words = prompt.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def chat_gpt_helper(client, prompt):
    """This function returns the response from OpenAI's Gpt-4 turbo model using the completions API."""
    try:
        resp = ''
        prompt_chunks = split_prompt(prompt)

        for chunk in prompt_chunks:
            for response_chunk in client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": chunk}],
                    max_tokens=4096,
                    n=1,
                    stop=None,
                    temperature=0,
                    stream=True  # Enable streaming
                ):
                content = response_chunk.choices[0].delta.content
                if content is not None:
                    yield content

    except Exception as e:
        log.info(e)
        yield str(e)
    
# @app.route('/chat', methods=['POST'])
# @limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
# def chat():
#     try:
#         loop = asyncio.new_event_loop()
#         docData = loop.run_until_complete(search_helper.getVectorStoreDocs(request))
#         webData = loop.run_until_complete(getWebExtract(request))
        
#         # Prepare the prompt
#         data = request.get_json()
#         query = data['q']
#         openai_key = data['openaiKey']
#         prompt = f"For this query: \n\n{query}, provide a detailed, accurate, and helpful answer based on the following data:\n\n{docData} and {webData}"
#         client = OpenAI(api_key=openai_key)
#         """
#         This streams the response from ChatGPT
#         """
#         return Response(stream_with_context(chat_gpt_helper(client, prompt)),
#                          mimetype='text/event-stream')
#     except Exception as e:
#         log.error(str(e))
#         return str(e)
    
# @app.route('/qna', methods=['POST'])
# @limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
# def qna():
#     try:
        
#         loop = asyncio.new_event_loop()
#         webData = loop.run_until_complete(getWebExtractLCFormat(request))
#         log.info(f"Web extarct LC format {webData}")
#         # Prepare the prompt
#         data = request.get_json()
#         query = data['q']
#         topk = data['topk']
#         count = data['count']
#         openai_key = data['openaiKey']
#         vectorstore = Pinecone.from_existing_index(index_name="structhub" , text_key='text',namespace='20', embedding=VertexAIEmbeddings("text-multilingual-embedding-preview-0409"))
#         llm = ChatOpenAI(model="gpt-4o", callbacks=callbacks, streaming=True, verbose=False, temperature=0.0, api_key=openai_key)
#         llm_mq = ChatOpenAI(model="gpt-4o", api_key=openai_key)
#         multiQueryPrompt = PromptTemplate(
#                     input_variables=["question"],
#                     template=f"""You are an AI language model assistant. Your task is 
#                 to generate 3 different versions of the given user 
#                 question to retrieve relevant documents from a vector  database. 
#                 By generating multiple perspectives on the user question, 
#                 your goal is to help the user overcome some of the limitations 
#                 of distance-based similarity search. Provide these alternative 
#                 questions separated by newlines. Original question: {query}
#                 """)

#         COMBINE_QUESTION_PROMPT_TEMPLATE = """You are a research agent and create detailed report in paragraph form. Use the following portion of a long document to see if any of the text is relevant to answer the question. 
#                     Return any relevant text in {language}.
#                     {context}
#                     Question: {question}
#                     Relevant text, if any, in {language}:"""

#         COMBINE_QUESTION_PROMPT = PromptTemplate(
#                         template=COMBINE_QUESTION_PROMPT_TEMPLATE, input_variables=[
#                             "context", "question", "language"]
#             )

#         COMBINE_PROMPT_TEMPLATE = f"""
#         # Instructions:
#         - You are a research agent and create detailed report in paragraph form with references from `Source` and `Page` section. 
#         - Include as much information possible. Do not try to summarize.
#         - Include statistics wherever available.
#         - Given the following extracted parts from one or multiple documents, and a question, create a final answer with references. 
#         - The reference must be from the `Source:` section of the extracted part.
#         - Refrence `Page` can be page number of document or a title of a page.
#         - Reference `Source` can be a URL of a webiste page or a docuemnt name.
#         - Always provide `Source` value. never miss this. It is very important.
#         - **You can only answer the question from information contained in the extracted parts below**, DO NOT use your prior knowledge.
#         - Never provide an answer without references.
#         - Never provide an answer without references.This is very important and must be adhered to.
#         - If you don't know the answer, just say that you don't know. Don't try to make up an answer.

#         =========
#         QUESTION: {{question}}
#         =========
#         {{summaries}}
#         =========
#         FINAL ANSWER in {{language}} :"""

#         COMBINE_PROMPT = PromptTemplate(
#             template=COMBINE_PROMPT_TEMPLATE, input_variables=[
#                 "summaries", "question"]
#         )

#         DOCUMENT_PROMPT = PromptTemplate(
#             template="Content: {page_content}\nSource: {source}\nPage: {page}",
#             input_variables=["page_content", "source", "page"],
#         )

#         # retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), llm=llm_mq, prompt=multiQueryPrompt, include_original=True)
#         retriever = vectorstore.as_retriever(
#                     search_kwargs={"k": topk})
#         docs = retriever.invoke(input=query)
#         docs = docs + webData
#         chain = qa_with_sources.load_qa_with_sources_chain(llm, chain_type="map_reduce", verbose=False,
#                         question_prompt=COMBINE_QUESTION_PROMPT,
#                         combine_prompt=COMBINE_PROMPT,
#                         document_prompt=DOCUMENT_PROMPT,
#                         return_intermediate_steps=False,
#                         token_max=4800)
#         def getAnswer():
#             result = chain.invoke({"input_documents": docs, "question": query, "language": "english"})
#             if result['output_text'] is not None:
#                 print(result)
#                 yield result['output_text']


#         return Response(stream_with_context(getAnswer()),
#                             mimetype='text/event-stream')
#     except Exception as e:
#         log.error(str(e))
#         return str(e)
# Function to split data into chunks
def split_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


# @app.route('/groqchat', methods=['POST'])
# @limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
# def groqchat():
#     try:
#         docData = asyncio.run(getVectorStoreDocs(request))
#         webData = asyncio.run(getWebExtract(request))
        
#         # Prepare the prompt
#         data = request.get_json()
#         query = data['q']
#         system = """
#             You are a helpful assistant.
#             Always respond to user's question with a JSON object with three keys:
#              - "response": This has the final generated answer. While generating answer always add numbered citations.
#              - "sources: sources key should be array of 'page' field as the page value from metadata section of context, it's `source` file name or URL which was used in generating final answer, and `citation` number.
#              - "followup_question".
#              Add only those `sources` from which citations are added to final answer. 
#             Use this data from web search {webdocs} and this is data from private knowledge store {kbdocs}, which always have source information which could be file page, page number and url.
#             """
#         human = "{question}"
#         # chain = prompt | chat
#         def getAnswer():
#             prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
#             chain = prompt | chat_stream
#             for chunk in chain.stream({"question": query, "webdocs": webData, "kbdocs": docData}):
#                 yield chunk.content
#         # response = chain.invoke({"question": query, "docs": docs})
#         # print(response)
#         # return response.content
#         # return Response(stream_with_context(getAnswer()),
#         #                  mimetype='text/event-stream')
#         return Response(stream_with_context(getAnswer()),
#                          mimetype='text/event-stream')
#     except Exception as e:
#         log.error(str(e))
#         return str(e)

@app.route('/geminichat', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def geminichat():
    try:
        data = request.get_json()
        query = data.get('q')
        sources = data.get('sources', [])
        topk = data.get('topk', 5)
        count = data.get('count', 5)

        docData = []
        webData = []

        if 'vstore' in sources:
            docData = asyncio.run(getVectorStoreDocs(request))
        if 'web' in sources:
            webData = asyncio.run(getWebExtract(request))

        system = """
            You are a helpful assistant.
            Always respond to the user's question with a JSON object containing three keys:
            - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation with 1.
            - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include the `text` field (the chunk of context), a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
            - `followup_question`: This key should contain a follow-up question relevant to the user's query.

            Make sure to only include `sources` from which citations are created. DO NOT include sources not used in generating the final answer.

            Use this data from web search {webdocs} and from the private knowledge store {kbdocs}, which always have source information including file page, page number, and URL.

            Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
            """
        human = "{question}"
        
        def getAnswer():
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
            chain = prompt | vertexchat_stream
            for chunk in chain.stream({"question": query, "webdocs": webData, "kbdocs": docData}):
                yield chunk.content

        return Response(stream_with_context(getAnswer()), mimetype='text/event-stream')
    except Exception as e:
        log.error(str(e))
        return str(e)


# @app.route('/anthropic', methods=['POST'])
# @limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
# def anthropic():
#     try:
#         data = request.get_json()
#         query = data.get('q')
#         sources = data.get('sources', [])
#         history = data.get('history', [])

#         docData = []
#         webData = []

#         if 'vstore' in sources:
#             docData = asyncio.run(getVectorStoreDocs(request))
#         if 'web' in sources:
#             webData = asyncio.run(getWebExtract(request))

#         raw_context = (
#             f"""
#                     You are a helpful assistant.
#                     Always respond to the user's question with a JSON object containing three keys:
#                     - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation with 1.
#                     - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include  a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
#                     - `followup_question`: This key should contain a follow-up question relevant to the user's query.

#                     Make sure to only include `sources` from which citations are created. DO NOT include sources not used in generating the final answer.
#                     DO NOT use your existing information and only use the information provided below to generate fial answer.
#                     Use this data from web search {webData} and from the private knowledge store {docData}, which always have source information including file page, page number, and URL.

#                     Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
#                     """
#         )

#         question = f"{query}"
#         context = SystemMessage(content=raw_context)
#         message = HumanMessage(content=question)
#         #Build message for topic
#         log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
#         pubsub_message = json.dumps({
#         "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
#         "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
#         "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
#         "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
#         "creditsUsed": 35
#         })
#         log.info(f"Averaged out credit usage for tokens: 25000")
#         log.info(f" Chargeable creadits: 35")
#         log.info(f"Message to topic: {pubsub_message}")
#         # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
#         google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=pubsub_message)
#         def getAnswer():
#             sync_response = anthropic_chat.stream([context, message])
#             for chunk in sync_response:
#                 log.info(chunk)
#                 yield chunk.content

#         return Response(stream_with_context(getAnswer()), mimetype='text/event-stream')
#     except Exception as e:
#         log.error(str(e))
#         raise Exception("There was an issue generating the answer. Please retry")

@app.route('/askme', methods=['POST'])
@limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
def askme():
    try:
        data = request.get_json()
        query = data.get('q')
        sources = data.get('sources', [])
        history = data.get('history', [])

        docData = []

        if 'vstore' in sources:
            docData = asyncio.run(getVectorStoreDocs(request))

        raw_context = (
            f"""
                    You are a helpful assistant.
                    Always respond to the user's question with a JSON object containing three keys:
                    - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation with 1. Make sure this section is in markdown format.
                    - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include  a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
                    - `followup_question`: This key should contain a follow-up question relevant to the user's query.

                    Make sure to only include `sources` from which citations are created. DO NOT include sources not used in generating the final answer.
                    DO NOT use your existing information and only use the information provided below to generate fial answer.
                    Use this data from  the private knowledge store {docData}, which always have source information including file page, page number, and URL.

                    Respond in JSON format. Do not add ```json at the beginning or ``` at the end of response. The output needs to only have JSON data, nothing else. THIS IS VERY IMPORTANT. Do not duplicate sources in the `sources` array.
                    """
        )

        question = f"{query}"
        #Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        pubsub_message = json.dumps({
        "subscription_id": getattr(request, 'tenant_data', {}).get('subscription_id', None),
        "user_id": getattr(request, 'tenant_data', {}).get('user_id', None),
        "keyName": getattr(request, 'tenant_data', {}).get('keyName', None),
        "project_id": getattr(request, 'tenant_data', {}).get('project_id', None),
        "creditsUsed": 35
        })
        log.info(f"Averaged out credit usage for tokens: 25000")
        log.info(f" Chargeable creadits: 35")
        log.info(f"Message to topic: {pubsub_message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(GCP_PROJECT_ID,GCP_CREDIT_USAGE_TOPIC, message=pubsub_message)
        def getAnswer():

            # class Source(BaseModel):
            #     citation: int
            #     page: float
            #     source: str

            # class EOLResponse(BaseModel):
            #     response: str
            #     sources: List[Source]
            #     followup_question: str

            # Generate the JSON schema from the Pydantic model
            # response_schema = {
            #     "required": [
            #         "response",
            #         "sources",
            #         "followup_question",
            #     ],
            #     "properties": {
            #         "response": {"type": "string"},
            #         "sources": {
            #             "type": "array",
            #             "items": {
            #                 "type": "object",
            #                 "properties": {
            #                     "citation": {"type": "integer"},
            #                     "page": {"type": "number"},
            #                     "source": {"type": "string"},
            #                 },
            #                 "required": ["citation", "page", "source"]
            #             }
            #         },
            #         "followup_question": {"type": "string"},
            #     },
            #     "type": "object",
            # }


            sync_response = google_genai_client.models.generate_content_stream(
                                 model='gemini-2.0-flash-exp',
                                 contents=question,
                                 config= {
                                 'system_instruction': raw_context,
                                 'temperature': 0.3
                                 }
                            )
            for chunk in sync_response:
                log.info(chunk)
                yield chunk.text

        return Response(stream_with_context(getAnswer()), mimetype='text/event-stream')
    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")


# @app.route('/chatmr', methods=['POST'])
# @limiter.limit(limit_value=lambda: getattr(request, 'tenant_data', {}).get('rate_limit', None),on_breach=default_error_responder)
# def chatmr():
#     try:
#         data = request.get_json()
#         query = data.get('q')
#         sources = data.get('sources', [])

#         docData = []
#         webData = []

#         if 'vstore' in sources:
#             docData = asyncio.run(getVectorStoreDocs(request))
#         if 'web' in sources:
#             webData = asyncio.run(getWebExtract(request))
#         log.info(f"Web Data {webData}")
#         log.info(f"Doc Data {docData}")
#         def chunk_data(data, chunk_size=10):
#             for i in range(0, len(data), chunk_size):
#                 yield data[i:i + chunk_size]

#         async def generate_partial_responses(query, combined_data):
#             tasks = []
#             log.info(f"Combined Data {combined_data}")
#             for doc_chunk, web_chunk in combined_data:
#                 raw_context = (
#                     f"""
#                             You are a helpful assistant.
#                             Always respond to the user's question with a JSON object containing three keys:
#                             - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation with 1.
#                             - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include the `text` field (the chunk of context), a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
#                             - `followup_question`: This key should contain a follow-up question relevant to the user's query.

#                             Make sure to only include `sources` from which citations are created. DO NOT include sources not used in generating the final answer.
#                             DO NOT use your existing information and only use the information provided below to generate final answer.
#                             Use this data from web search {web_chunk} and from the private knowledge store {doc_chunk}, which always have source information including file page, page number, and URL.

#                             Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
#                             """
#                 )
#                 question = f"{query}"
#                 context = SystemMessage(content=raw_context)
#                 message = HumanMessage(content=question)
#                 tasks.append(anthropic_chat.ainvoke([context, message]))

#             return await asyncio.gather(*tasks)

#         async def main():
#             combined_data = list(zip(chunk_data(docData), chunk_data(webData)))
            
#             partial_responses = await generate_partial_responses(query, combined_data)
#             log.info(f"Partial Responses {partial_responses}")
#             final_response_content = "".join([response['response'] for response in partial_responses])
#             final_sources = []
#             for response in partial_responses:
#                 final_sources.extend(response['sources'])

#             final_context = (
#                 f"""
#                     You are a helpful assistant.
#                     Always respond to the user's question with a JSON object containing three keys:
#                     - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation with 1.
#                     - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include the `text` field (the chunk of context), a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
#                     - `followup_question`: This key should contain a follow-up question relevant to the user's query.

#                     Make sure to only include `sources` from which citations are created. DO NOT include sources not used in generating the final answer.
#                     DO NOT use your existing information and only use the information provided below to generate final answer.
#                     Use this data from previous responses {final_response_content}, which have source information including file page, page number, and URL.

#                     Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
#                 """
#             )
#             context = SystemMessage(content=final_context)
#             message = HumanMessage(content=query)

#             return context, message
        
#         # Check if there's an existing event loop
#         try:
#             loop = asyncio.get_running_loop()
#             context, message = loop.run_until_complete(main())
#         except RuntimeError:  # No event loop is running
#             context, message = asyncio.run(main())
#         log.info(f"Final context message {context} {message}")

#         def stream_response():
#             sync_response = anthropic_chat.stream([context, message])
#             for chunk in sync_response:
#                 yield chunk.content


#         return Response(stream_with_context(stream_response()), mimetype='text/event-stream')
    
#     except Exception as e:
#         log.error(str(e))
#         return str(e)

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