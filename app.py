import os
import io
import re
import json
import time
import sys
import ssl
import signal
import asyncio
import concurrent.futures
from datetime import datetime

from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from flask_limiter import Limiter, RequestLimit
from werkzeug.utils import secure_filename
from urllib.parse import urlencode

from google import genai
from google.genai import types
from vertexai.preview.language_models import TextEmbeddingModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pinecone import PineconeAsyncio as PC  # or from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

# BM25 + sparse vector utilities
from shared.bm25 import (
    tokenize_document,
    compute_bm25_sparse_vector,
    get_project_vocab_stats,
    get_project_alpha,
    hybrid_score_norm,
)

from shared import (
    file_processor,
    google_auth,
    security,
    google_pub_sub,
    search_helper,
    psql,
)

from shared.logging_config import log

# For robust date parsing
from dateutil.parser import parse as dateutil_parse, ParserError

app = Flask(__name__)
CORS(app, origins="*")
app.debug = True

# ====================================================================================
# ENV
# ====================================================================================
REDIS_HOST = os.environ.get("REDIS_HOST")
REDIS_PORT = os.environ.get("REDIS_PORT")
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")
SECRET_KEY = os.environ.get("SECRET_KEY")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER")

SERVER_URL = os.environ.get("SERVER_URL", "")
WEBSEARCH_SERVER_URL = os.environ.get("WEBSEARCH_SERVER_URL", "")

app.config["UPLOAD_FOLDER"] = UPLOADS_FOLDER

# Create the Bearer token for Tika
bearer_token = google_auth.impersonated_id_token(
    serverurl=SERVER_URL
).json()["token"]

# Vertex chat with some basic safety
vertexchat_stream = ChatVertexAI(
    model="gemini-2.0-flash-exp",
    response_mime_type="application/json",
    safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    },
)

# The standard GenAI client
google_genai_client = genai.Client(
    vertexai=True, 
    project=GCP_PROJECT_ID, 
    location="us-central1"
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


# Rate-limiting
def default_error_responder(request_limit: RequestLimit):
    return Response(status=429)

limiter = Limiter(
    key_func=lambda: getattr(request, "tenant_data", {}).get("subscription_id", None),
    app=app,
    storage_uri=f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/xtract",
    storage_options={"socket_connect_timeout": 30},
    strategy="moving-window",
    on_breach=default_error_responder,
)

# ====================================================================================
# UTILS
# ====================================================================================
def get_event_loop():
    """Retrieve or create the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

async def get_google_embedding(queries):
    """Async function to get embeddings for a list of strings."""
    embedder_name = "text-multilingual-embedding-preview-0409"
    model = TextEmbeddingModel.from_pretrained(embedder_name)
    embeddings_list = model.get_embeddings(queries)
    embeddings = [embedding.values for embedding in embeddings_list]
    return embeddings

def get_sparse_vector(query: str, project_id: str, vocab_stats: dict, max_terms: int) -> dict:
    """Compute BM25 sparse vector for a single query string."""
    tokens = tokenize_document(query)
    return compute_bm25_sparse_vector(
        tokens, project_id, vocab_stats, max_terms=max_terms
    )

def generate_query_keywords(query: str, history: list) -> list[str]:
    """
    Synchronous helper that uses the GenAI client to get an array
    of keywords and variations from (query+history). Returns a list of strings.
    """
    combined_text = f"History:\n{json.dumps(history)}\nQuery:\n{query}"

    prompt_text = f"""
You have the following user query (and chat history).
Generate a robust list of 10-50 keywords or short phrases relevant to this request,
including synonyms, abbreviations, expansions, and possible variations the user might ask.
Return strictly valid JSON: an array of strings, no extra keys.

Text to analyze:
{json.dumps(combined_text, ensure_ascii=False)}
"""
    text_part = types.Part.from_text(text=prompt_text)

    try:
        response = google_genai_client.models.generate_content(
            model="gemini-2.0-flash-lite-001",
            contents=[types.Content(role="user", parts=[text_part])],
            config={
                "temperature": 0.5,
                "top_p": 0.95,
                "max_output_tokens": 1024,
                "response_modalities": ["TEXT"],
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
        )
        text_result = response.text
        keywords = json.loads(text_result)
        if not isinstance(keywords, list):
            return []
        keywords = [k for k in keywords if isinstance(k, str)]
        return keywords
    except Exception as e:
        log.warning(f"[generate_query_keywords] LLM call failed: {str(e)}")
        return []

def convert_dates_for_metadata_filter(filter_dict: dict, metadata_keys: list) -> dict:
    """
    Given the metadata filter (a dictionary) and the list of metadata_keys from DB,
    convert values to a Unix timestamp if:
      - The key exists in the filter,
      - Its type in metadata_keys is defined as 'Date', and
      - The value is a string that can be parsed as a date.
    If conversion fails, leave the original value.
    """
    db_key_map = {}
    for mk in metadata_keys:
        k = mk.get("key")
        t = mk.get("type")
        if k and t:
            db_key_map[k] = t

    for key, value in filter_dict.items():
        if key in db_key_map and db_key_map[key] == "Date":
            if isinstance(value, str):
                try:
                    dt = dateutil_parse(value)
                    filter_dict[key] = int(dt.timestamp())
                except (ParserError, ValueError):
                    pass
    return filter_dict

# ====================================================================================
# HEALTH
# ====================================================================================
@app.route("/health", methods=["GET"], endpoint="health_check")
def health_check():
    return json.dumps({"status": "ok"})

# ====================================================================================
# MAIN SEARCH ROUTINE (async)
# ====================================================================================
async def getVectorStoreDocs(request):
    """
    Updated approach (async):
      1) Parse user data
      2) Generate multi-queries from user query+history
      3) In parallel, also generate metadata filter + array of keywords
      4) Perform 3 pinecone searches:
         (a) metadata filter + auto_topics filter
         (b) metadata filter only
         (c) multi-query approach
      5) Merge results in order, removing duplicates
      6) Apply top-K or token limit
      7) Return final docs as JSON
    """
    data = request.get_json()
    query = data["q"]
    history = data.get("history", [])
    user_id = getattr(request, "tenant_data", {}).get("user_id", None)
    project_id = getattr(request, "tenant_data", {}).get("project_id", None)
    if not user_id or not project_id:
        raise Exception("Invalid request — user_id or project_id not found")

    # 1) Generate multi-queries from the user query (synchronous LLM call)
    multi_query_prompt = search_helper.create_multi_query_prompt(history, query)
    sync_response = google_genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=multi_query_prompt,
        config={"temperature": 0.3},
    )
    multi_query_resp = sync_response.text
    all_lines = [q.strip() for q in multi_query_resp.split("\n") if q.strip()]

    try:
        last_line_json = json.loads(all_lines[-1])
        if isinstance(last_line_json, list):
            keywords_extracted = last_line_json
            search_queries = all_lines[:-1] + keywords_extracted
        else:
            search_queries = all_lines
    except:
        search_queries = all_lines

    if not search_queries:
        search_queries = [query]

    first_query = search_queries[0]

    # 2) Generate metadata filter from user query
    project_row = psql.get_project_details(project_id)
    metadata_prompt = project_row.get("metadataPrompt", "")
    metadata_keys = project_row.get("metadataKeys", [])
    if isinstance(metadata_keys, str):
        try:
            metadata_keys = json.loads(metadata_keys)
        except:
            metadata_keys = []

    metadata_filter = search_helper.generate_metadata_filter_advanced(
        first_query=first_query,
        metadata_prompt=metadata_prompt,
        metadata_keys=metadata_keys,
        google_genai_client=google_genai_client,
    )
    # Once metadata filter is created, convert all Date types to Unix timestamp.
    if metadata_filter:
        metadata_filter = convert_dates_for_metadata_filter(metadata_filter, metadata_keys)

    # 3) Generate array of keywords from user query+history for auto_topics
    auto_topics_list = await asyncio.to_thread(generate_query_keywords, query, history)

    # 4) Create embeddings for the multi-query approach
    embeddings_list = await get_google_embedding(queries=search_queries)

    # 5) Prepare Pinecone
    pc = PC(api_key=PINECONE_API_KEY)
    indexModel = await pc.describe_index(name=PINECONE_INDEX_NAME)
    index = pc.IndexAsyncio(host=indexModel.index.host)
    vocab_stats = get_project_vocab_stats(project_id)
    alpha = get_project_alpha(vocab_stats)

    async def do_pinecone_multi_query(dense_vec, query_str):
        sparse_vec = get_sparse_vector(query_str, project_id, vocab_stats, 300)
        dv, sv = hybrid_score_norm(dense=dense_vec, sparse=sparse_vec, alpha=alpha)
        response = await index.query(
            namespace=project_id,
            vector=dv,
            sparse_vector=sv,
            top_k=1000,
            include_values=False,
            include_metadata=True,
        )
        return response.to_dict().get("matches", [])

    async def do_metadata_filter_query(filter_obj, vec):
        response = await index.query(
            namespace=project_id,
            vector=vec,
            top_k=1000,
            include_values=False,
            include_metadata=True,
            filter=filter_obj if filter_obj else {},
        )
        return response.to_dict().get("matches", [])

    # Build the two filters
    combined_filter_1 = None
    if metadata_filter and auto_topics_list:
        combined_filter_1 = {
            "$and": [
                metadata_filter,
                {"auto_topics": {"$in": auto_topics_list}},
            ]
        }
    elif auto_topics_list:
        combined_filter_1 = {"auto_topics": {"$in": auto_topics_list}}
    else:
        combined_filter_1 = metadata_filter

    combined_filter_2 = metadata_filter if metadata_filter else {}

    async def run_multi_queries():
        all_hits = []
        tasks = []
        for i, sq in enumerate(search_queries):
            tasks.append(do_pinecone_multi_query(embeddings_list[i], sq))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, list):
                all_hits.extend(r)
            else:
                log.warning(f"[multi-query error] {str(r)}")
        return all_hits

    tasks = []
    run_filters = []
    if combined_filter_1:
        run_filters.append(("filter_1", do_metadata_filter_query(
            combined_filter_1, embeddings_list[0] if embeddings_list else []
        )))
    if combined_filter_2:
        run_filters.append(("filter_2", do_metadata_filter_query(
            combined_filter_2, embeddings_list[0] if embeddings_list else []
        )))
    run_filters.append(("multi_query", run_multi_queries()))

    results_map = {}
    tasks_labels = [asyncio.create_task(coro, name=label) for (label, coro) in run_filters]
    done, pending = await asyncio.wait(tasks_labels, return_when=asyncio.ALL_COMPLETED)
    for d in done:
        label = d.get_name()
        try:
            results_map[label] = d.result()
        except Exception as e:
            log.warning(f"Task {label} error: {str(e)}")
            results_map[label] = []

    results_1 = results_map.get("filter_1", [])
    results_2 = results_map.get("filter_2", [])
    results_3 = results_map.get("multi_query", [])

    # Logging the filters and document counts
    log.info(f"Combined Filter 1: {combined_filter_1}")
    log.info(f"Combined Filter 2: {combined_filter_2}")
    log.info(f"Documents fetched for Filter 1: {len(results_1)}")
    log.info(f"Documents fetched for Filter 2: {len(results_2)}")
    log.info(f"Documents fetched for Multi-Query: {len(results_3)}")

    unique_docs = {}
    final_merged = []

    for doc in results_1:
        doc_id = doc["id"]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = True
            final_merged.append(doc)
    for doc in results_2:
        doc_id = doc["id"]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = True
            final_merged.append(doc)
    for doc in results_3:
        doc_id = doc["id"]
        if doc_id not in unique_docs:
            unique_docs[doc_id] = True
            final_merged.append(doc)

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    max_tokens = 200000
    topk_docs = []
    total_tokens = 0
    for doc in final_merged:
        if len(topk_docs) >= 1000:
            break
        doc_str = json.dumps(doc)
        doc_tokens = len(enc.encode(doc_str))
        if total_tokens + doc_tokens > max_tokens:
            break
        topk_docs.append(doc)
        total_tokens += doc_tokens

    async def publish_credits():
        count = len(topk_docs)
        message = json.dumps(
            {
                "subscription_id": getattr(request, "tenant_data", {}).get("subscription_id", None),
                "user_id": user_id,
                "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                "project_id": project_id,
                "creditsUsed": count * 0.2,
            }
        )
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
        )

    asyncio.create_task(publish_credits())
    return json.dumps(topk_docs), 200

# ====================================================================================
# ASKME ENDPOINT
# ====================================================================================
@app.route("/askme", methods=["POST"])
@limiter.limit(
    limit_value=lambda: getattr(request, "tenant_data", {}).get("rate_limit", None),
    on_breach=default_error_responder,
)
def askme():
    """
    Endpoint that:
     1) Runs getVectorStoreDocs() to gather doc data
     2) Streams out an LLM chat response as JSON
    """
    try:
        data = request.get_json()
        query = data.get("q")
        sources = data.get("sources", [])
        history = data.get("history", [])
        docData = "[]"
        if "vstore" in sources:
            loop = get_event_loop()
            docData, _status_code = loop.run_until_complete(getVectorStoreDocs(request))
        raw_context = f"""
        You are a helpful assistant.
        Always respond to the user's question with a JSON object containing three keys:
        - `response`: A detailed, accurate answer in Markdown, with citations like [1], [2], etc.  
        - `sources`: An array with the text or partial text for each source used. Each source must include a `citation`, `page`, `source`.
        - `followup_question`: A relevant, first-person question.

        Only use sources from the private knowledge store below. If you do not find relevant sources, just respond with best effort from the data provided.

        Private knowledge store: {docData}

        Always respond only in valid JSON. No extra code fences.
        """
        question = f"{query}"
        pubsub_message = json.dumps(
            {
                "subscription_id": getattr(request, "tenant_data", {}).get("subscription_id", None),
                "user_id": getattr(request, "tenant_data", {}).get("user_id", None),
                "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                "project_id": getattr(request, "tenant_data", {}).get("project_id", None),
                "creditsUsed": 35,
            }
        )
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=pubsub_message
        )
        def getAnswer():
            sync_response = google_genai_client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=question,
                config={"system_instruction": raw_context, "temperature": 0.3},
            )
            for chunk in sync_response:
                log.info(chunk)
                yield chunk.text
        response = Response(stream_with_context(getAnswer()), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"
        response.headers["Transfer-Encoding"] = "chunked"
        return response
    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")

# ====================================================================================
# SHUTDOWN HANDLER
# ====================================================================================
def shutdown_handler(signal_int: int, frame):
    log.info(f"Caught Signal {signal.strsignal(signal_int)}")
    sys.exit(0)

# ====================================================================================
# MAIN
# ====================================================================================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
