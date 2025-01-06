import base64
import json
import logging
import os
import sys
import re
import time
import hashlib
import asyncio
import uuid
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, request
from PIL import Image, UnidentifiedImageError

# Atlassian + Pinecone
from atlassian.confluence import Confluence
from requests.exceptions import HTTPError

# langchain / text-splitting
from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vertex AI embedding
from vertexai.preview.language_models import TextEmbeddingModel

# Shared code imports
from shared.logging_config import log
from shared import psql
from pinecone import Pinecone

# OPTIONAL: If you want to do encryption
# from Crypto.Cipher import AES
# from Crypto.Util.Padding import pad, unpad
# from Crypto.Random import get_random_bytes

app = Flask(__name__)

# Initialize connection to Pinecone
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('PINECONE_INDEX_NAME')
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)
########################################################################
# MONKEY PATCHING
########################################################################

# 1. Patch Confluence HTTP error handling
original_raise_for_status = Confluence.raise_for_status


def custom_raise_for_status(self, response):
    """
    Checks the response for an error status and raises an exception
    with the error message provided by the server
    """
    if 400 <= response.status_code < 600 and response.status_code != 404:
        try:
            j = response.json()
            error_msg = j.get("message", "Unknown error from Confluence")
        except Exception as e:
            log.warning(f"Couldn't parse Confluence error JSON: {e}")
            # fallback
            response.raise_for_status()
        else:
            raise HTTPError(error_msg, response=response)


Confluence.raise_for_status = custom_raise_for_status

# 2. Patch Pillow large-file bomb check
_decompression_bomb_check_orig = Image._decompression_bomb_check


def custom_decompression_bomb_check(size):
    try:
        _decompression_bomb_check_orig(size)
    except Image.DecompressionBombError as e:
        log.info(f"Monkey patched: File size exceeded: {str(e)}")
        return False


setattr(Image, "_decompression_bomb_check", custom_decompression_bomb_check)

########################################################################
# HELPER FUNCTIONS
########################################################################


def get_event_loop():
    """
    Safely retrieve or create an asyncio event loop.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def generate_md5_hash(*args):
    """
    Generates an MD5 hash from the JSON-serialized arguments.
    """
    serialized_args = [json.dumps(arg) for arg in args]
    combined_string = "|".join(serialized_args)
    md5_hash = hashlib.md5(combined_string.encode("utf-8")).hexdigest()
    return md5_hash


async def get_google_embedding(queries, model_name="text-multilingual-embedding-preview-0409"):
    """
    Uses Vertex AI to embed a list of text queries.
    """
    log.info(f"Getting embeddings for {len(queries)} queries using model '{model_name}'.")
    model = TextEmbeddingModel.from_pretrained(model_name)
    embeddings_list = model.get_embeddings(texts=queries, auto_truncate=True)
    embeddings = [embedding.values for embedding in embeddings_list]
    log.info("Embeddings fetched successfully.")
    return embeddings


########################################################################
# MAIN CONFLUENCE INGEST FUNCTION
########################################################################

def process_confluence_requests(
    confluence_url: str,
    confluence_user: str,
    confluence_token: str,
    parent_page_id: str,
    data_source_id: str,
    project_id: str,
    namespace: str,
    subscription_id: str = None,
    embedding_model_name: str = "text-multilingual-embedding-preview-0409",
    last_sync_time: datetime = None
):
    """
    Combines:
    - Confluence child-page traversal
    - Google Vertex AI embeddings
    - Pinecone ingestion
    - psql usage (e.g. checking credits, updating status)
    - Incremental sync logic based on last_sync_time and existing files in DB.
    """
    log.info("Starting Confluence ingestion with incremental sync logic...")
    log.debug(f"Parameters => confluence_url: {confluence_url}, user: {confluence_user}, "
              f"parent_page_id: {parent_page_id}, project_id: {project_id}, subscription_id: {subscription_id}, "
              f"namespace: {namespace}, last_sync_time: {last_sync_time}")

    # (1) Optional: check subscription credits via psql
    if subscription_id:
        remaining_credits = psql.get_remaining_credits(subscription_id)
        log.info(f"Remaining credits for subscription {subscription_id}: {remaining_credits}")
        if remaining_credits <= 0:
            log.error(f"No credits left for subscription {subscription_id}. Aborting.")
            return

    # (2) Initialize Confluence client
    try:
        confluence = Confluence(
            url=confluence_url,
            username=confluence_user,
            password=confluence_token,
        )
        log.info("Confluence client initialized.")
    except Exception as e:
        log.exception("Failed to initialize Confluence client:")
        return

    # (3) Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    
    # (5) Recursive function to gather child page IDs
    def get_all_child_ids(page_id):
        """
        Traverses Confluence child pages from a given page_id.
        """
        result = []
        stack = [page_id]
        processed = set()

        log.debug(f"Starting to collect child IDs for page {page_id}")

        while stack:
            current_id = stack.pop()
            if current_id in processed:
                continue
            processed.add(current_id)
            result.append(current_id)

            log.debug(f"Processing page {current_id}. Total so far: {len(result)}")

            child_ids = confluence.get_child_id_list(page_id=current_id)
            new_child_ids = [cid for cid in child_ids if cid not in processed]
            stack.extend(new_child_ids)

        log.debug(f"Finished collecting child IDs for {page_id}. Total pages: {len(result)}")
        return result

    # (5a) Collect child page IDs
    try:
        child_ids = get_all_child_ids(parent_page_id)
        log.info(f"Discovered child page IDs for {parent_page_id}: {child_ids}")
    except Exception as e:
        log.exception("Failed to fetch child page IDs:")
        return
    
   # (A) If last_sync_time is empty => full sync
    # (B) If last_sync_time is present => partial sync
    #    We fetch existing DB records + do "only updated" logic
    existing_files_by_page_id = {}  # e.g. { <page_id>: <file_record_dict> }

    if last_sync_time:
        log.info(f"Incremental sync: lastSyncTime = {last_sync_time}")
        # Convert `last_sync_time` to datetime if needed
        if isinstance(last_sync_time, str):
            try:
                last_sync_time_dt = datetime.fromisoformat(last_sync_time)
            except ValueError:
                log.warning(f"Invalid lastSyncTime format: {last_sync_time}, ignoring.")
                last_sync_time_dt = None
        else:
            last_sync_time_dt = last_sync_time
    else:
        log.info("No lastSyncTime provided => full sync of all pages.")
        last_sync_time_dt = None

    # (A) Fetch existing files in DB for this dataSourceId => so we can see what we have
    existing_files = psql.fetch_files_by_data_source_id(data_source_id)
    for ef in existing_files:
        # e.g. in DB, we store pageID in some column or we need to parse from file_key
        # If you stored "pageId" in "file_key" or separate column, adapt accordingly
        # Suppose we have a column "pageId" in the DB or "key" in metadata
        page_id_db = ef.get("pageId")  # or some other approach
        if page_id_db:
            existing_files_by_page_id[page_id_db] = ef

    # (B) We'll identify pages to ingest and files to delete
    #  1) For each confluence child_id, check if "page" is new or updated => ingest
    #  2) For each existing file => if not in child_ids => delete
    child_batch_size = 50
    processed_pages_count = 0

    # (B1) Collect pages that are in DB but not in child_ids => remove them
    # e.g. we have existing_files_by_page_id keys that are not in child_ids
    db_page_ids = set(existing_files_by_page_id.keys())
    confluence_page_ids = set(child_ids)
    removed_page_ids = db_page_ids - confluence_page_ids
    log.info(f"Pages in DB but not in Confluence => to be removed: {removed_page_ids}")

    for rp_id in removed_page_ids:
        file_record = existing_files_by_page_id[rp_id]
        file_id_to_delete = file_record.get("id")
        if file_id_to_delete:
            # 1) Delete from Pinecone
            #    typically we have "fileId" or something => build the vector ID
            #    e.g. fileRecord might have a column "id" => we used it in vector "file_id#project_id"
            #    If the DB 'id' is "myFileId" => vector_id = "myFileId#<projectId>"
            vector_id = f"{file_id_to_delete}#{project_id}"
            try:
                index.delete(ids=[vector_id], namespace=namespace)
                log.info(f"Deleted Pinecone vector {vector_id}")
            except Exception as e:
                log.exception(f"Error deleting vector {vector_id} from Pinecone:")

            # 2) Delete from PostgreSQL
            psql.delete_file_by_id(file_id_to_delete)
            log.info(f"Deleted DB file id={file_id_to_delete} for pageId={rp_id}")

    # (B2) For each child_id in child_ids => check lastModified from Confluence
    # We can do e.g. confluence.get_page_by_id(..., expand='version')
    for i in range(0, len(child_ids), child_batch_size):
        batch_page_ids = child_ids[i : i + child_batch_size]
        log.info(
            f"Processing batch {i // child_batch_size + 1} with {len(batch_page_ids)} pages out of {len(child_ids)} total."
        )

        # We'll fetch the docs with ConfluenceLoader or we do a direct confluence.get_page_by_id for lastModified
        # If the page is older than lastSyncTime => skip
        # If no last_sync_time => process anyway

        # Could do a direct approach:
        updated_batch_page_ids = []
        for page_id in batch_page_ids:
            try:
                page_info = confluence.get_page_by_id(page_id, expand="version")
                # page_info['version']['when'] e.g. "2023-08-24T20:58:47.000Z"
                page_last_modified_str = page_info.get("version", {}).get("when")
                if not page_last_modified_str:
                    updated_batch_page_ids.append(page_id)
                    continue

                # Convert string to datetime
                # Confluence typically returns e.g. "2023-08-24T20:58:47.000Z"
                page_last_modified_dt = datetime.fromisoformat(
                    page_last_modified_str.replace("Z", "+00:00")
                )

                if last_sync_time_dt:
                    if page_last_modified_dt > last_sync_time_dt:
                        # page is newer => process
                        updated_batch_page_ids.append(page_id)
                    else:
                        log.info(f"Page {page_id} not changed since last sync => skipping.")
                else:
                    # no last sync => always process
                    updated_batch_page_ids.append(page_id)

            except Exception as e:
                log.exception(f"Failed to retrieve page info for {page_id}, skipping:")
                continue

        if not updated_batch_page_ids:
            log.info("No updated pages in this batch => skipping ingestion.")
            continue

        # (C) Load raw documents from Confluence for the updated pages
        loader = ConfluenceLoader(
            url=confluence_url,
            username=confluence_user,
            api_key=confluence_token,
            include_attachments=True,
            limit=50,
            max_pages=5000,
            page_ids=updated_batch_page_ids,
        )

        try:
            documents = loader.load()
        except Exception as e:
            log.exception("Error loading documents from Confluence:")
            continue

        # (7a) Split into chunks
        docs = text_splitter.split_documents(documents)
        log.info(f"Total doc-chunks after split: {len(docs)}")

        # Prepare Pinecone upsert
        pinecone_vectors = []
        batch_size = 20
        local_counter = 0

        for idx, doc in enumerate(docs, start=1):
            page_id = doc.metadata.get("id", "unknown_id")
            page_content = doc.page_content or ""

            sub_for_hash = subscription_id if subscription_id else "no_subscription"
            file_id = generate_md5_hash(sub_for_hash, project_id, page_id)

            now_dt = datetime.now()

            # Check if file is new or existing in DB
            # e.g. existing_files_by_page_id.get(page_id)
            # If the file_id is the same (some logic?), we can update or re-insert
            # We'll do the same psql.add_new_file flow (or psql.update_file) if you prefer
            try:
                psql.add_new_file(
                    file_id,
                    status="processing",
                    data_source_id=data_source_id,
                    created_at=now_dt,
                    updated_at=doc.metadata.get("when", now_dt),
                    name=doc.metadata.get("source", "ConfluencePage"),
                    file_type="url",
                    file_key=f"{sub_for_hash}-{project_id}-{file_id}",
                    page_count=1,
                    upload_url="None",
                    source_type="confluence",
                    project_id=project_id,
                )
            except Exception as e:
                log.exception(f"Failed to add new file record in psql for doc {page_id}:")

            # Create embedding
            try:
                embedding_result = asyncio.run(get_google_embedding([page_content], embedding_model_name))
                embedding = embedding_result[0]
            except Exception as e:
                log.exception(f"Error creating embeddings for doc {page_id}:")
                continue

            # Prepare vector
            vector_id = f"{file_id}#{project_id}"
            vector_metadata = {
                "content": page_content,
                "sourceType": "Confluence",
                "dataSourceId": data_source_id,
                "pageId": page_id,
                "title": doc.metadata.get("title", ""),
                "lastModified": str(doc.metadata.get("when", now_dt)),
            }
            pinecone_vectors.append((vector_id, embedding, vector_metadata))
            local_counter += 1

            if (local_counter % batch_size == 0) or (idx == len(docs)):
                try:
                    log.info(f"Upserting {len(pinecone_vectors)} vectors to Pinecone (batch upsert).")
                    index.upsert(vectors=pinecone_vectors, namespace=namespace)
                except Exception as e:
                    log.exception("Error upserting to Pinecone:")
                pinecone_vectors.clear()

            processed_pages_count += 1

        log.info(f"Finished processing updated doc-chunks from pages: {updated_batch_page_ids}")

    # (10) Final status update in psql
    if processed_pages_count > 0:
        try:
            psql.update_file_status(
                file_id=file_id,  # last processed
                status="processed",
                page_count=processed_pages_count,
                updated_at=datetime.now(),
            )
        except Exception as e:
            log.exception("Error updating final file status in psql:")
    else:
        log.info("No pages processed or no docs found; skipping final status update.")

    log.info("Confluence ingestion (incremental sync) completed successfully.")


########################################################################
# PUB/SUB ENDPOINT FOR CLOUD RUN
########################################################################

@app.route("/", methods=["POST"])
def event_handler():
    """
    Cloud Run endpoint to handle cloud run function request.
    Th ecloud run function triggered from pub/sub sends a message from GCP topic 'confluence-topic-{env}',
    and that message triggers this container.
    """
    data = request.get_json()
    log.info(f"Event Data: {data}")
    DATA_SOURCE_CONFIG = os.environ.get("DATA_SOURCE_CONFIG")
    event_data = json.loads(json.loads(DATA_SOURCE_CONFIG).get("event_data"))
    log.info(f"Cloud Run Job received data: {event_data}") 
    # Extract parameters from the message
    confluence_url = event_data.get("confluenceUrl")
    confluence_user = event_data.get("confluenceUser")
    confluence_token = event_data.get("confluenceToken")
    parent_page_id = event_data.get("confluenceParent")
    last_sync_time = event_data.get("lastSyncTime")
    project_id = event_data.get("projectId")
    namespace = project_id  # For example, you can set Pinecone namespace = projectId
    data_source_id = event_data.get("id")

    # get project details ( and subscription_id)
    project_details = psql.get_project_details(project_id=project_id)
    if project_details:
        # Proceed with processing the project details
        log.info(f"Project Details: {project_details}")
        subscription_id = project_details.get("subscriptionId")
        # Validate required fields
        required_fields = {
            "confluenceURL": confluence_url,
            "confluenceUser": confluence_user,
            "confluenceToken": confluence_token,
            "parentPageId": parent_page_id,
            "projectId": project_id,
            "dataSourceId": data_source_id,
            "subscriptionId": subscription_id,
        }
        missing = [k for k, v in required_fields.items() if not v]
        if missing:
            log.error(f"Missing required fields: {missing}")
            return (f"Bad Request: Missing fields {missing}", 400)

        try:
            process_confluence_requests(
                confluence_url=confluence_url,
                confluence_user=confluence_user,
                confluence_token=confluence_token,
                parent_page_id=parent_page_id,
                data_source_id=data_source_id,
                project_id=project_id,
                namespace=namespace,
                subscription_id=subscription_id,
                embedding_model_name="text-multilingual-embedding-preview-0409",
                last_sync_time=last_sync_time
            )
            return ("OK", 200)
        except Exception as e:
            log.exception("Error in process_confluence_requests:")
            return ("Internal Server Error", 500)


if __name__ == "__main__":
    # For local testing/debugging:
    # You can run: `python main.py` then do e.g. a local request with Cloud Run or a similar approach
    # In Cloud Run, Gunicorn usually calls app, not this block.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
