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
    hybrid_score_norm,
)
from langchain_core.outputs import LLMResult
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

callbacks = [FinalStreamingStdOutCallbackHandler()]
from vertexai.preview.language_models import TextEmbeddingModel
from urllib.parse import urlencode
import json
from shared import (
    file_processor,
    google_auth,
    security,
    google_pub_sub,
    search_helper,
    psql,
)
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


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_sparse_vector(
    query: str, project_id: str, vocab_stats: dict, max_terms: int
) -> dict:

    # 1) Tokenize,
    tokens = tokenize_document(query)
    return compute_bm25_sparse_vector(
        tokens, project_id, vocab_stats, max_terms=max_terms
    )


@app.route("/health", methods=["GET"], endpoint="health_check")
def health_check():
    return json.dumps({"status": "ok"})


async def getVectorStoreDocs(request):
    try:
        # 1) Parse inbound data
        data = request.get_json()
        query = data["q"]
        history = data.get("history", [])
        user_id = getattr(request, "tenant_data", {}).get("user_id", None)
        project_id = getattr(request, "tenant_data", {}).get("project_id", None)
        if not user_id or not project_id:
            raise Exception("Invalid request — user_id or project_id not found")

        # 2) Generate multiple queries via an LLM (depends on user query)
        multi_query_prompt = search_helper.create_multi_query_prompt(history, query)
        sync_response = google_genai_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=multi_query_prompt,
            config={"temperature": 0.2},
        )
        multi_query_resp = sync_response.text
        all_queries = [q.strip() for q in multi_query_resp.split("\n") if q.strip()]

        # last line is a possible JSON of keywords
        try:
            keywords = json.loads(all_queries[-1])
            if not isinstance(keywords, list):
                keywords = [all_queries[-1]]
            search_queries = all_queries[:-1] + keywords
        except:
            # fallback if last line is not JSON
            search_queries = all_queries
            keywords = [search_queries[-1]]

        # 3) Generate metadata filter (depends on the first query)
        first_query = search_queries[0]
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

        # 4) Embeddings (depends on multi-query generation)
        embeddings_list = await get_google_embedding(queries=search_queries)

        # 5) Prepare Pinecone
        pc = PC(api_key=PINECONE_API_KEY)
        index = pc.Index(name=PINECONE_INDEX_NAME)
        vocab_stats = get_project_vocab_stats(project_id)

        # Our hybrid query function
        def do_pinecone_query(dense_vec, query_str):
            sparse_vec = get_sparse_vector(query_str, project_id, vocab_stats, 300)
            alpha = get_project_alpha(vocab_stats)
            dv, sv = hybrid_score_norm(dense=dense_vec, sparse=sparse_vec, alpha=alpha)
            response = index.query(
                namespace=project_id,
                vector=dv,
                sparse_vector=sv,
                top_k=1000,
                include_values=False,
                include_metadata=True,
            )
            return response.to_dict().get("matches", [])

        # 6) Run Pinecone searches in parallel
        #    (both the multi-query search AND the metadata-filtered query)
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # If we have a metadata filter, submit that query in parallel
            if metadata_filter:
                futures.append(
                    executor.submit(
                        index.query,
                        namespace=project_id,
                        vector=embeddings_list[0],
                        top_k=1000,
                        include_values=False,
                        include_metadata=True,
                        filter=metadata_filter,
                    )
                )

            # Submit each multi-query to Pinecone
            for i, sq in enumerate(search_queries):
                futures.append(
                    executor.submit(do_pinecone_query, embeddings_list[i], sq)
                )

            # Collect all results
            all_matches = []
            meta_results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    # If it's from a metadata-filter query, it returns
                    # a pinecone QueryResponse, not a list of dicts
                    if isinstance(result, dict) and "matches" in result:
                        meta_results.extend(result["matches"])
                    else:
                        all_matches.extend(result)
                except Exception as exc:
                    log.error(f"Query concurrency error: {exc}")

        # 7) Merge results
        all_matches.extend(meta_results)
        unique_docs = {}
        for doc in all_matches:
            doc_id = doc["id"]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc

        final_docs = list(unique_docs.values())

        # 8) Limit docs by token size
        enc = tiktoken.get_encoding("cl100k_base")
        max_tokens = 200000
        topk_docs = []
        total_tokens = 0
        # Re-add meta_results first if you want them on top
        meta_docs_on_top = [m for m in meta_results if "metadata" in m]
        doc_list = meta_docs_on_top + final_docs
        for doc in doc_list:
            if len(topk_docs) >= 1000:
                break
            doc_str = json.dumps(doc)
            doc_tokens = len(enc.encode(doc_str))
            if total_tokens + doc_tokens > max_tokens:
                break
            topk_docs.append(doc)
            total_tokens += doc_tokens

        # 9) Publish usage metrics asynchronously
        #    So returning the final result is not blocked
        loop = get_event_loop()

        async def publish_credits():
            count = len(topk_docs)
            message = json.dumps(
                {
                    "subscription_id": getattr(request, "tenant_data", {}).get(
                        "subscription_id", None
                    ),
                    "user_id": user_id,
                    "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                    "project_id": project_id,
                    "creditsUsed": count * 0.2,
                }
            )
            google_pub_sub.publish_messages_with_retry_settings(
                GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
            )

        # Schedule the publishing to run in the background
        loop.create_task(publish_credits())

        # 10) Return JSON
        return json.dumps(topk_docs), 200

    except Exception as e:
        log.error(str(e))
        raise Exception("There was an issue generating the answer. Please retry")


def split_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


@app.route("/askme", methods=["POST"])
@limiter.limit(
    limit_value=lambda: getattr(request, "tenant_data", {}).get("rate_limit", None),
    on_breach=default_error_responder,
)
def askme():
    try:
        data = request.get_json()
        query = data.get("q")
        sources = data.get("sources", [])
        history = data.get("history", [])

        docData = []

        if "vstore" in sources:
            docData = asyncio.run(getVectorStoreDocs(request))

        raw_context = f"""
                You are a helpful assistant.
                Always respond to the user's question with a JSON object containing three keys:
                - `response`: This key should have the final generated answer. Ensure the answer includes citations in the form of reference numbers (e.g., [1], [2]). Always start citation numbering from 1. Make sure this section is in markdown format.
                - `sources`: This key should be an array of the original chunks of context used in generating the answer. Each source should include a `citation` field (the reference number), a `page` field (the page value), and the `source` field (the file name or URL). Each source should appear only once in this array.
                - `followup_question`: This key should contain a follow-up question relevant to the user's query, phrased in the **first person**, as if the user is asking the question themselves. Do not use "Would you like" or "Do you want"; instead, structure the question directly as the user’s request.

                Make sure to only include `sources` from which citations are created. **DO NOT** include sources that were not used in generating the final answer.
                **DO NOT** use any prior knowledge; rely **only** on the provided information.
                Use this data from the private knowledge store {docData}, which always includes source information such as file pages, page numbers, and URLs.
                Always use tags in metadata section if available to pick values when generatng answers. 
                DO NOT try to sumamrize or shorten the answer. Be as descriptive as possible.
                Create answers in table format wherever possible. make the formatting as user friendly as possible.
                Respond in JSON format. **Do not add** ```json at the beginning or ``` at the end of the response. The output must contain only JSON data, nothing else. **THIS IS VERY IMPORTANT.** Do not duplicate sources in the `sources` array.
                """

        question = f"{query}"
        # Build message for topic
        log.info(f"tenant data {getattr(request, 'tenant_data', {})}")
        pubsub_message = json.dumps(
            {
                "subscription_id": getattr(request, "tenant_data", {}).get(
                    "subscription_id", None
                ),
                "user_id": getattr(request, "tenant_data", {}).get("user_id", None),
                "keyName": getattr(request, "tenant_data", {}).get("keyName", None),
                "project_id": getattr(request, "tenant_data", {}).get(
                    "project_id", None
                ),
                "creditsUsed": 35,
            }
        )
        log.info(f"Averaged out credit usage for tokens: 25000")
        log.info(f" Chargeable creadits: 35")
        log.info(f"Message to topic: {pubsub_message}")
        # topic_headers = {"Authorization": f"Bearer {bearer_token}"}
        google_pub_sub.publish_messages_with_retry_settings(
            GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=pubsub_message
        )

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
                model="gemini-2.0-flash-exp",
                contents=question,
                config={"system_instruction": raw_context, "temperature": 0.3},
            )
            for chunk in sync_response:
                log.info(chunk)
                yield chunk.text

        return Response(stream_with_context(getAnswer()), mimetype="text/event-stream")
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
