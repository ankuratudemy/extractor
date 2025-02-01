import os
import io
import re
from google import genai
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter, RequestLimit
from werkzeug.utils import secure_filename
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pinecone import Pinecone as PC
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

# BM25 + sparse vector utilities (with no spaCy usage)
from shared.bm25 import (
    tokenize_document,
    compute_bm25_sparse_vector,
    get_project_vocab_stats,
    get_project_alpha,
    hybrid_score_norm
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
import concurrent.futures

sys.path.append("../")
from shared.logging_config import log

app = Flask(__name__)
CORS(app, origins="*")
app.debug = True

# Assuming you have Redis connection details
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
SECRET_KEY = os.environ.get("SECRET_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
# Other env specific varibales:

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER")

oFileExtMap = {
    "application/octet-stream": "use_extension",  # use file extension if content type is octet-stream
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
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template": "dotx",
    "application/vnd.ms-word.document.macroenabled.12": "docm",
    "application/vnd.ms-word.template.macroenabled.12": "dotm",
    "application/xml": "xml",
    "application/msword": "doc",
    "application/vnd.ms-word.document.macroenabled.12": "docm",
    "application/vnd.ms-word.template.macroenabled.12": "dotm",
    "application/rtf": "rtf",
    "application/wordperfect": "wpd",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template": "xltx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsm",
    "application/vnd.ms-excel.template.macroenabled.12": "xltm",
    "application/vnd.corelqpw": "qpw",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow": "ppsx",
    "application/vnd.openxmlformats-officedocument.presentationml.slide": "ppmx",
    "application/vnd.openxmlformats-officedocument.presentationml.template": "potx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.12": "ppsm",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12": "pptm",
    "application/vnd.ms-powerpoint.addin.macroenabled.12": "ppam",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.12": "ppsm",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12": "pptm",
    "application/vnd.ms-powerpoint.addin.macroenabled.12": "ppam",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.ms-powerpoint.slideshow": "pps",
    "application/vnd.ms-powerpoint.presentation": "ppt",
    "application/vnd.ms-powerpoint.addin": "ppa",
    # Email formats
    "message/rfc822": "eml",  # EML format
    "application/vnd.ms-outlook": "msg",  # MSG format
    "application/mbox": "mbox",  # MBOX format
    "application/vnd.ms-outlook": "pst",  # PST format
    "application/ost": "ost",  # OST format
    "application/emlx": "emlx",  # EMLX format
    "application/dbx": "dbx",  # DBX format
    "application/dat": "dat",  # Windows Mail (.dat) format
    # Image formats
    "image/jpeg": "jpg",  # JPEG format
    "image/png": "png",  # PNG format
    "image/gif": "gif",  # GIF format
    "image/tiff": "tiff",  # TIFF format
    "image/bmp": "bmp",  # BMP format
}

anthropic_chat = ChatAnthropicVertex(
    model_name="claude-3-5-sonnet@20240620",
    project=GCP_PROJECT_ID,
    location="us-east5",
    max_tokens=4096,
    temperature=0.1,
)

chat_stream = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    streaming=True,
    # api_key="" # Optional if not set as an environment variable
)
chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    model_kwargs={"response_format": {"type": "json_object"}},
    # api_key="" # Optional if not set as an environment variable
)
vertexchat = ChatVertexAI(
    model="gemini-1.5-pro",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)
vertexchat_stream = ChatVertexAI(
    model="gemini-2.0-flash-exp",
    response_mime_type="application/json",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)

google_genai_client = genai.Client(
    vertexai=True, project=GCP_PROJECT_ID, location="us-central1"
)


@app.before_request
def before_request():
    log.info(f"Request endpoint: {request.endpoint}")
    protected_endpoints = [
        "extract",
        "search",
        "chat",
        "groqchat",
        "geminichat",
        "anthropic",
        "serp",
        "webextract",
        "qna",
        "chatmr",
        "askme",
    ]
    if request.endpoint in protected_endpoints and request.method != "OPTIONS":
        api_key_header = request.headers.get("API-KEY")
        valid = security.api_key_required(api_key_header)

        def generate():
            yield "No credits left to process request!"

        if not valid and (
            request.endpoint == "anthropic" or request.endpoint == "askme"
        ):
            return Response(
                stream_with_context(generate()), mimetype="text/event-stream"
            )
        if not valid:
            return Response(status=401)


def default_error_responder(request_limit: RequestLimit):
    # return jsonify({"message": f'rate Limit Exceeded: {request_limit}'}), 429
    return Response(status=429)


limiter = Limiter(
    key_func=lambda: getattr(request, "tenant_data", {}).get("subscription_id", None),
    app=app,
    storage_uri=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/xtract",
    storage_options={"socket_connect_timeout": 30},
    strategy="moving-window",  # or "moving-window"
    on_breach=default_error_responder,
)

app.config["UPLOAD_FOLDER"] = UPLOADS_FOLDER
SERVER_URL = f"https://{os.environ.get('SERVER_URL')}/tika"
WEBSEARCH_SERVER_URL = f"https://{os.environ.get('WEBSEARCH_SERVER_URL')}/search"


# Create the ID token
bearer_token = google_auth.impersonated_id_token(
    serverurl=os.environ.get("SERVER_URL")
).json()["token"]
websearch_bearer_token = google_auth.impersonated_id_token(
    serverurl=os.environ.get("WEBSEARCH_SERVER_URL")
).json()["token"]


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_sparse_vector(query: str, project_id: str, vocab_stats: dict, max_terms: int) -> dict:

    # 1) Tokenize, 
    tokens = tokenize_document(query)
    return compute_bm25_sparse_vector(
        tokens, project_id, vocab_stats, max_terms=max_terms
    )


@app.route("/health", methods=["GET"], endpoint="health_check")
def health_check():
    return json.dumps({"status": "ok"})


@app.route("/extract", methods=["POST"])
@limiter.limit(
    limit_value=lambda: getattr(request, "tenant_data", {}).get("rate_limit", None),
    on_breach=default_error_responder,
)
def extract():
    lang_value = request.form.get("lang", "")
    ocr_value = request.form.get("ocr", "")
    out = request.form.get("out_format", "")

    # # Define mapping for lang_value
    # lang_mapping = {
    #     'hindi': 'hin',
    #     # Add more mappings as needed
    # }

    # Define mapping for lang_value
    out_format_mapping = {
        "text": "text/plain",
        "xml": "application/xml",
        "html": "text/html",
        "json": "application/json",
        # Add more mappings as needed
    }

    # Define default values
    default_ocr_strategy = "auto"
    default_out_format = "text/plain"

    # Set values based on conditions
    x_tika_ocr_language = lang_value
    x_tika_pdf_ocr_strategy = (
        "no_ocr"
        if ocr_value.lower() == "false"
        else ("ocr_only" if ocr_value.lower() == "true" else default_ocr_strategy)
    )
    x_tika_accept = out_format_mapping.get(out, default_out_format)

    # Now you can use these values in your headers
    headers = {
        "X-Tika-PDFOcrStrategy": x_tika_pdf_ocr_strategy,
        "Accept": x_tika_accept,
    }

    # Add 'X-Tika-OCRLanguage' header only if value is not empty
    if x_tika_ocr_language:
        headers["X-Tika-OCRLanguage"] = x_tika_ocr_language

    # Access a specific header
    content_type_header = request.headers.get("Content-Type")
    log.info(f"\nContent-Type Header: {content_type_header}")

    contentType = "application/pdf"
    uploaded_file = request.files["file"]

    if uploaded_file:
        num_pages = 0
        # Save the uploaded file to a temporary location
        filename = secure_filename(uploaded_file.filename)
        temp_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        log.info(uploaded_file.content_type)

        # Reverse mapping of content types to file extensions
        reverse_file_ext_map = {v: k for k, v in oFileExtMap.items()}

        if uploaded_file.content_type not in oFileExtMap:
            log.info("Invalid file extension")
            return "Unsupported file format.", 400

        file_extension = oFileExtMap[uploaded_file.content_type]

        if file_extension == "use_extension":
            file_extension = os.path.splitext(filename)[1][1:].lower()

        log.info(file_extension)
        log.info(filename)

        if file_extension == "pdf":
            # Read the PDF file directly from memory
            pdf_data = uploaded_file.read()

            # Measure the time taken to split the PDF
            start_time = time.time()
            pages, num_pages = file_processor.split_pdf(pdf_data)
            split_time = time.time() - start_time
            # log.info(f"Time taken to split the PDF: {split_time * 1000} ms")

        elif file_extension in ["csv", "xls", "xltm", "xltx", "xlsx", "tsv", "ots"]:
            pages = file_processor.split_excel(uploaded_file.read())
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, "")

        elif file_extension in [
            "eml",
            "msg",
            "pst",
            "ost",
            "mbox",
            "dbx",
            "dat",
            "emlx",
        ]:
            pages = [("1", io.BytesIO(uploaded_file.read()))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, "")

        elif file_extension in ["jpg", "jpeg", "png", "gif", "tiff", "bmp", "html"]:
            pages = [("1", io.BytesIO(uploaded_file.read()))]
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, "")

        elif file_extension in ["ods"]:
            pages = file_processor.split_ods(uploaded_file.read())
            num_pages = len(pages)
            contentType = reverse_file_ext_map.get(file_extension, "")

        elif file_extension in [
            "docx",
            "pdf",
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
            "pptm",
            "ppam",
            "ppt",
            "pps",
            "ppt",
            "ppa",
            "rtf",
        ]:
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
                log.error("Conversion to PDF failed")
                return "Conversion to PDF failed.", 400
        else:
            log.error("Unsupported file format")
            return "Unsupported file format.", 400

        # # Get the number of available CPUs
        # num_cpus = multiprocessing.cpu_count()
        # log.info(num_cpus)

        # Append Header value
        headers["Content-Type"] = contentType
        headers["Authorization"] = f"Bearer {bearer_token}"

        loop = get_event_loop()
        results = loop.run_until_complete(process_pages_async(pages, headers))

        # Build the JSON output using mapped_results
        json_output = []
        chargeable_credits = 0
        for result, page_num in results:
            if result.strip():  # Check if the text is not empty after stripping
                chargeable_credits += 1
            page_obj = {"page": page_num, "text": result.strip()}
            json_output.append(page_obj)

        json_string = json.dumps(json_output, indent=4)
        log.info(f"Extraction successful for file: {filename}")

        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        # send CREDIT USAGE TO TOPIC 'structhub-credit-usage'

        # Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps(
            {
                "subscription_id": getattr(request, "tenant_data", {}).get(
                    "subscription_id", None
                ),
                "user_id": getattr(request, "tenant_data", {}).get("user_id", None),
                "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                "project_id": getattr(request, "tenant_data", {}).get(
                    "project_id", None
                ),
                "creditsUsed": chargeable_credits,
            }
        )
        log.info(f"Number of pages processed: {num_pages}")
        log.info(f" Chargeable creadits: {chargeable_credits}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
        )
        return json_string, 200, {"Content-Type": "application/json; charset=utf-8"}

    else:
        log.error("No file uploaded")
        return "No file uploaded.", 400


def convert_to_pdf(file_path, file_extension):
    import subprocess

    try:
        # Specify the output PDF file path
        pdf_file_path = os.path.splitext(file_path)[0] + ".pdf"

        # Convert the file to PDF using LibreOffice
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

            # Delete the output PDF file
            os.remove(pdf_file_path)

            return pdf_data
        else:
            # log.error('Conversion to PDF failed: Output file not found')
            return None
    except subprocess.CalledProcessError as e:
        log.error(f"Conversion to PDF failed: {str(e)}")
        return None
    except Exception as e:
        log.error(f"Error during PDF conversion: {str(e)}")
        return None


async def process_pages_async(pages, headers):
    url = SERVER_URL
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_put_request(session, url, page_data, page_num, headers)
            for page_num, page_data in pages
        ]
        results = await asyncio.gather(*tasks)

    # Map results to their corresponding page numbers
    mapped_results = [(result, page_num) for (result, page_num) in results]

    return mapped_results


async def async_put_request(session, url, payload, page_num, headers, max_retries=10):
    retries = 0

    while retries < max_retries:
        try:
            payload_copy = io.BytesIO(payload.getvalue())
            async with session.put(
                url,
                data=payload_copy,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 429:
                    log.info(
                        f"Retrying request for page {page_num}, Retry #{retries + 1}"
                    )
                    retries += 1
                    await asyncio.sleep(1)  # You may adjust the sleep duration
                    continue  # Retry the request
                content = await response.read()  # Read the content of the response
                text_content = content.decode(
                    "utf-8", errors="ignore"
                )  # Decode bytes to Unicode text, ignoring errors
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


@app.route("/search", methods=["POST"])
@limiter.limit(
    limit_value=lambda: getattr(request, "tenant_data", {}).get("rate_limit", None),
    on_breach=default_error_responder,
)
def search():
    try:
        userId = getattr(request, "tenant_data", {}).get("user_id", None)
        projectId = getattr(request, "tenant_data", {}).get("project_id", None)
        namespace = projectId
        log.info(f"user_id {userId}")
        log.info(f"project id to search(namespace) {projectId}")
        if not userId:
            return "invalid request", 400

        # Check if request body contains the required parameters
        if not request.is_json:
            return "invalid request", 400

        data = request.get_json()
        log.info(data)
        if "topk" not in data or "query" not in data:
            missing_params = []
            if "topk" not in data:
                missing_params.append("topk")
            if "query" not in data:
                missing_params.append("query")
            return f"Missing required parameters: {', '.join(missing_params)}", 400

        topk = data["topk"]
        query = data["query"]
        log.info(f"query: {query} topk: {topk}")
    except Exception as e:
        log.info(str(e))
        return "validation: Something went wrong", 500

    try:
        pc = PC(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
    except Exception as e:
        log.info(str(e))
        return "vectorDB: Something went wrong", 500

    try:
        # Get google embeddings:
        embeddings = asyncio.run(get_google_embedding(queries=[query]))
    except Exception as e:
        log.error(str(e))
        return "vectorDB Search: Something went wrong", 500

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
        doc_list = [
            {
                "source": match["metadata"]["source"],
                "page": match["metadata"]["page"],
                "text": match["metadata"]["text"],
            }
            for match in docs.to_dict()["matches"]
        ]
        count = len(doc_list)
        final_response = {"count": count, "data": doc_list}
        json_string = json.dumps(final_response, indent=4)
        # Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        message = json.dumps(
            {
                "subscription_id": getattr(request, "tenant_data", {}).get(
                    "subscription_id", None
                ),
                "user_id": getattr(request, "tenant_data", {}).get("user_id", None),
                "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                "project_id": getattr(request, "tenant_data", {}).get(
                    "project_id", None
                ),
                "creditsUsed": 0.2,
            }
        )
        log.info(f"Number of pages processed: {count}")
        log.info(f"Message to topic: {message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
        )
        return json_string, 200, {"Content-Type": "application/json; charset=utf-8"}
    except Exception as e:
        log.error(str(e))
        return "Query index: Something went wrong", 500


async def async_get_request(session, url, params, headers=None, max_retries=10):
    retries = 0
    query_string = urlencode(params)

    while retries < max_retries:
        try:
            async with session.get(
                f"{url}?{query_string}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
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

                return json_content, params.get("pageno", 1)

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


async def getVectorStoreDocs(request):
    try:
        user_id = getattr(request, "tenant_data", {}).get("user_id", None)
        project_id = getattr(request, "tenant_data", {}).get("project_id", None)
        namespace = project_id
        log.info(f"user_id {user_id}")
        log.info(f"project id (namespace) {project_id}")

        if not user_id:
            raise Exception("invalid request")

        if not request.is_json:
            raise Exception("invalid request")

        data = request.get_json()
        log.info(data)
        if "topk" not in data or "q" not in data:
            missing_params = []
            if "topk" not in data:
                missing_params.append("topk")
            if "q" not in data:
                missing_params.append("q")
            return f"Missing required parameters: {', '.join(missing_params)}", 400

        topk = data["topk"]
        query = data["q"]
        history = data.get("history", [])
        log.info(f"query: {query} | topk: {topk}")

    except Exception as e:
        log.info(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # ---------------------------------------------------------------------
    # 1) Generate Multi-Queries using a small LLM prompt
    # ---------------------------------------------------------------------
    try:
        multi_query_prompt = search_helper.create_multi_query_prompt(history, query)
        # e.g. Anthropics or Vertex or any LLM:
        # multi_query_resp = anthropic_chat.invoke(input=multi_query_prompt).content

        # For demonstration, using the Google GenAI (Vertex) snippet
        sync_response = google_genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=multi_query_prompt,
            config={"temperature": 0.5},
        )
        multi_query_resp = sync_response.text

        # The last line or portion typically contains keywords as JSON or similar
        all_queries = [q.strip() for q in multi_query_resp.split("\n") if q.strip()]
        log.info(f"Generated Queries: {all_queries}")

        # We assume the last line is some JSON array of keywords or fallback
        keyword_query = all_queries[-1]
        try:
            keywords = json.loads(
                keyword_query
            )  # e.g. ["enterprise data", "compliance"]
            if not isinstance(keywords, list):
                keywords = [keyword_query]
        except:
            keywords = [keyword_query]

        search_queries = all_queries[:-1] + keywords

    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # ---------------------------------------------------------------------
    # 2) Batch-embed all queries for DENSE vectors
    # ---------------------------------------------------------------------
    try:
        # get_google_embedding returns an array of embeddings, one per query
        embeddings_list = await get_google_embedding(queries=search_queries)
    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

    # ---------------------------------------------------------------------
    # 3) For each query: build a sparse vector & do a Pinecone query
    #    We parallelize this step to speed up multiple queries
    # ---------------------------------------------------------------------
    try:
        pc = PC(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
    except Exception as e:
        log.info(str(e))
        raise Exception("There was an issue generating the answer. Please retry")
    # 2) Get BM25 stats
    vocab_stats = get_project_vocab_stats(project_id)
    def do_pinecone_query(dense_vec, query_str, vocab_stats):
        """
        Helper function that:
         1. Builds a sparse vector (BM25) from query_str
         2. Performs a Pinecone hybrid query using both dense & sparse
        """
        # Build the sparse vector with your BM25 logic
        # Adjust max_terms as needed
        sparse_vec = get_sparse_vector(query_str, project_id, vocab_stats, 300)

        # get project alpha
        alpha = get_project_alpha(vocab_stats)

        #Get weghted dense, sparse vectors
        dv,sv = hybrid_score_norm(dense=dense_vec, sparse=sparse_vec, alpha=alpha)
        # Example format of sparse_vec:
        # {
        #    "indices": [...],
        #    "values": [...]
        # }
        # Optionally, you can pass a "score_by" param for controlling how Pinecone merges
        # the dense + sparse dot products. E.g. 'dot_product', 'cosine', or use 'sum'.
        response = index.query(
            namespace=str(namespace),
            vector=dv,
            sparse_vector=sv,
            top_k=topk,
            include_values=False,
            include_metadata=True,
            # score_by="dot_product" or "cosine", etc.
        )
        return response.to_dict().get("matches", [])

    # We'll run the queries in parallel via ThreadPoolExecutor
    all_matches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_query = {}
        for i, sq in enumerate(search_queries):
            dense_vec = embeddings_list[i]
            future = executor.submit(do_pinecone_query, dense_vec, sq, vocab_stats)
            future_to_query[future] = sq

        for future in concurrent.futures.as_completed(future_to_query):
            sq = future_to_query[future]
            try:
                result_matches = future.result()
                all_matches.extend(result_matches)
            except Exception as exc:
                log.error(f"Query {sq} generated an exception: {exc}")

    # ---------------------------------------------------------------------
    # 4) Deduplicate and reorder docs using your re-ordering logic
    # ---------------------------------------------------------------------
    # De-duplicate by doc ID
    unique_docs = {}
    for doc in all_matches:
        doc_id = doc["id"]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc

    final_docs = list(unique_docs.values())

    # Convert final_docs to your required format
    doc_list = []
    for match in final_docs:
        doc_list.append(
            {
                "source": match["metadata"]["source"],
                "page": match["metadata"]["page"],
                "text": match["metadata"]["text"],
            }
        )

    # Optional: reorder based on regex/keyword matches
    doc_list = await search_helper.reorder_docs(doc_list, keywords)

    # ---------------------------------------------------------------------
    # 5) Respect token limits; pick up to topk docs without exceeding ~50k tokens
    # ---------------------------------------------------------------------
    enc = tiktoken.get_encoding("cl100k_base")
    max_tokens = 50000

    topk_docs = []
    total_tokens = 0
    for doc in doc_list:
        if len(topk_docs) >= topk:
            break
        doc_str = json.dumps(doc)
        doc_tokens = len(enc.encode(doc_str))
        if total_tokens + doc_tokens > max_tokens:
            break
        topk_docs.append(doc)
        total_tokens += doc_tokens

    log.info(f"Final topk_docs count: {len(topk_docs)}, total_tokens: {total_tokens}")

    # ---------------------------------------------------------------------
    # 6) Publish usage metrics & return JSON
    # ---------------------------------------------------------------------
    count = len(topk_docs)
    message = json.dumps(
        {
            "subscription_id": getattr(request, "tenant_data", {}).get(
                "subscription_id", None
            ),
            "user_id": user_id,
            "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
            "project_id": project_id,
            "creditsUsed": count * 0.2,  # or however you're tracking usage
        }
    )
    google_pub_sub.publish_messages_with_retry_settings(
        GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
    )

    return json.dumps(topk_docs), 200


def split_prompt(prompt, max_tokens=20000):
    """Splits the prompt into chunks that fit within the max token limit."""
    words = prompt.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def split_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]

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
                    Use this data from the private knowledge store {docData} and consider the conversation history {history}, which always have source information including file page, page number, and URLs for different sources.
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
