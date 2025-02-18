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

# PIL
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
from shared import psql, google_pub_sub
from pinecone import Pinecone

# BM25 + sparse vector utilities (with no spaCy usage)
from shared.bm25 import (
    tokenize_document,
    publish_partial_bm25_update,
    compute_bm25_sparse_vector,
    get_project_vocab_stats
)

# ----------------------------------------------------------------------------
# ENV VARIABLES
# ----------------------------------------------------------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB")
GCP_CREDIT_USAGE_TOPIC = os.environ.get("GCP_CREDIT_USAGE_TOPIC")
UPLOADS_FOLDER = os.environ.get("UPLOADS_FOLDER", "/tmp/uploads")  # default

######################################################################
# MONKEY PATCHING
######################################################################

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


######################################################################
# HELPER FUNCTIONS
######################################################################

def get_sparse_vector(file_texts: list[str], project_id: str, max_terms: int) -> dict:
    """
    - Combine all file text into one big string.
    - Publish partial update once so aggregator merges the entire file's token freq.
    - Fetch and return the updated vocab stats from Firestore.
    """
    # 1) Combine
    combined_text = " ".join(file_texts)

    # 2) Tokenize
    tokens = tokenize_document(combined_text)

    # 3) Publish partial update => aggregator merges term freq
    publish_partial_bm25_update(project_id, tokens, is_new_doc=True)

    # 4) Get BM25 stats from Firestore => { "N":..., "avgdl":..., "vocab": { term-> {df, tf} } }
    vocab_stats = get_project_vocab_stats(project_id)

    return compute_bm25_sparse_vector(tokens, project_id, vocab_stats, max_terms=max_terms)




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


######################################################################
# MAIN CONFLUENCE INGEST FUNCTION
######################################################################

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
    last_sync_time=None  # Could be string or datetime
):
    """
    Combines:
    - Confluence child-page traversal
    - Google Vertex AI embeddings
    - Pinecone ingestion
    - psql usage (e.g. checking credits, updating status)
    - Incremental sync logic based on last_sync_time and existing files in DB.
    Additionally processes new pages that are present in Confluence but not in the DB.
    """
    log.info("Starting Confluence ingestion with incremental sync logic...")
    log.debug(
        f"Parameters => confluence_url: {confluence_url}, user: {confluence_user}, "
        f"parent_page_id: {parent_page_id}, project_id: {project_id}, subscription_id: {subscription_id}, "
        f"namespace: {namespace}, last_sync_time: {last_sync_time}"
    )

    # Initialize Pinecone
    api_key = os.environ.get('PINECONE_API_KEY')
    index_name = os.environ.get('PINECONE_INDEX_NAME')
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

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
        log.error("Failed to initialize Confluence client:")
        psql.update_data_source_by_id(data_source_id, status="processing")
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
        log.error("Failed to fetch child page IDs:")
        return
    
    # Ensure the parent_page_id is included
    if parent_page_id not in child_ids:
        log.info(f"Parent page ID {parent_page_id} not in child_ids. Adding it explicitly.")
        child_ids.append(parent_page_id)

    # (5b) Convert lastSyncTime => UTC datetime (if it's a string)
    last_sync_time_dt = None
    if last_sync_time:
        if isinstance(last_sync_time, str):
            # e.g. "2025-01-07T05:22:05.339Z"
            try:
                # parse using strptime
                last_sync_time_dt = datetime.strptime(last_sync_time, "%Y-%m-%dT%H:%M:%S.%fZ")
                # mark it as UTC
                last_sync_time_dt = last_sync_time_dt.replace(tzinfo=timezone.utc)
                log.info(f"Parsed lastSyncTime as => {last_sync_time_dt.isoformat()}")
            except ValueError:
                log.warning(f"Invalid lastSyncTime format: {last_sync_time}, ignoring.")
                last_sync_time_dt = None
        else:
            # If last_sync_time was already a datetime obj
            last_sync_time_dt = last_sync_time
            # If it’s naive, we might want to set tzinfo=timezone.utc
            if last_sync_time_dt.tzinfo is None:
                last_sync_time_dt = last_sync_time_dt.replace(tzinfo=timezone.utc)
    else:
        log.info("No lastSyncTime provided => full sync of all pages.")

    # (A) Fetch existing files in DB for this dataSourceId
    existing_files = psql.fetch_files_by_data_source_id(data_source_id)

    # Build a set of all file_ids in DB (these are the MD5-hash-based IDs or 'key')
    db_file_keys = set()
    file_map = {}  # Map from file_id -> file record
    for ef in existing_files:
        # In your table, "id" is the MD5 you created. 
        file_id_db = ef["id"]
        db_file_keys.add(file_id_db)
        file_map[file_id_db] = ef

    # (B) Identify which files should remain (based on Confluence child page IDs)
    confluence_file_keys = set()

    # If you have a subscription_id, replicate the logic you used for hashing:
    sub_for_hash = subscription_id if subscription_id else "no_subscription"
    log.info(f"sub_for_hash: {sub_for_hash}")
    page_id_to_file_key = {}  # Map page_id to file_key for easy lookup
    for c_page_id in child_ids:
        log.info(f"c_page_id: {c_page_id}")
        c_file_key = generate_md5_hash(sub_for_hash, project_id, data_source_id, c_page_id)
        confluence_file_keys.add(c_file_key)
        page_id_to_file_key[c_page_id] = c_file_key

    # (C) Determine which file keys in DB are NOT present in Confluence -> remove them
    removed_file_keys = db_file_keys - confluence_file_keys
    log.info(f"Files in DB but not in Confluence => to be removed: {removed_file_keys}")

    for r_key in removed_file_keys:
        file_record = file_map[r_key]
        # Step 1) Remove from Pinecone
        vector_id = f"{data_source_id}#{r_key}#{project_id}"
        try:
            index.delete(ids=[vector_id], namespace=namespace)
            log.info(f"Deleted Pinecone vector {vector_id}")
        except Exception as e:
            log.error(f"Error deleting vector {vector_id} from Pinecone:")

        # Step 2) Remove from PostgreSQL
        try:
            psql.delete_file_by_id(r_key)  # r_key = MD5-based ID
            log.info(f"Deleted DB File with ID={r_key}")
        except Exception as e:
            log.error(f"Error deleting DB file {r_key}:")

    # (D) Identify New and Updated Pages
    new_page_ids = []
    updated_page_ids = []

    for page_id in child_ids:
        file_key = page_id_to_file_key.get(page_id)
        if not file_key:
            log.warning(f"No file key found for page_id {page_id}, skipping.")
            continue

        if file_key not in db_file_keys:
            # New Page
            new_page_ids.append(page_id)
            log.info(f"Identified new page: {page_id}")
        else:
            # Existing Page - check if updated
            try:
                page_info = confluence.get_page_by_id(page_id, expand="version")
                page_last_modified_str = page_info.get("version", {}).get("when")
                if not page_last_modified_str:
                    # If no version info, treat as updated
                    updated_page_ids.append(page_id)
                    continue

                # parse Confluence's time, e.g. "2025-01-07T05:22:05.339Z"
                page_last_modified_dt = datetime.strptime(
                    page_last_modified_str, "%Y-%m-%dT%H:%M:%S.%fZ"
                ).replace(tzinfo=timezone.utc)

                if last_sync_time_dt:
                    if page_last_modified_dt > last_sync_time_dt:
                        updated_page_ids.append(page_id)
                    else:
                        log.info(f"Page {page_id} not changed since last sync => skipping.")
                else:
                    # full sync
                    updated_page_ids.append(page_id)

            except Exception as e:
                log.error(f"Failed to retrieve page info for {page_id}, skipping:")
                continue

    log.info(f"Total new pages to process: {len(new_page_ids)}")
    log.info(f"Total updated pages to process: {len(updated_page_ids)}")

    # Combine new and updated page IDs
    pages_to_process = new_page_ids + updated_page_ids

    if not pages_to_process:
        log.info("No new or updated pages to process. Ingestion complete.")
        return

    # Process pages in batches
    page_batch_size = 50
    for i in range(0, len(pages_to_process), page_batch_size):
        batch_page_ids = pages_to_process[i : i + page_batch_size]
        log.info(
            f"Processing batch {i // page_batch_size + 1} with {len(batch_page_ids)} "
            f"pages out of {len(pages_to_process)} total."
        )

        # (C) Load raw documents from Confluence
        loader = ConfluenceLoader(
            url=confluence_url,
            username=confluence_user,
            api_key=confluence_token,
            include_attachments=True,
            limit=50,
            max_pages=5000,
            page_ids=batch_page_ids,
        )

        try:
            documents = loader.load()
            log.info(f"Loaded {len(documents)} documents from Confluence.")
        except Exception as e:
            log.error("Error loading documents from Confluence:")
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
            file_id = generate_md5_hash(sub_for_hash, project_id, data_source_id, page_id)

            now_dt = datetime.now(timezone.utc)

            # Upsert / Insert into psql
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
                log.info(f"Added new file record for file_id: {file_id}")
            except Exception as e:
                log.error(f"Failed to add new file record in psql for doc {page_id}:")
                continue

            # Create embedding
            try:
                embedding_result = asyncio.run(
                    get_google_embedding([page_content], embedding_model_name)
                )
                embedding = embedding_result[0]
            except Exception as e:
                log.error(f"Error creating embeddings for doc {page_id}:")
                continue

            # Prepare vector
            sparse_values = get_sparse_values(page_content, namespace)
            vector_id = f"{data_source_id}#{file_id}#{project_id}"
            vector_metadata = {
                "text": page_content,
                "sourceType": "Confluence",
                "dataSourceId": data_source_id,
                "pageId": page_id,
                "title": doc.metadata.get("title", ""),
                "lastModified": str(doc.metadata.get("when", now_dt)),
                "source": doc.metadata.get("source", "ConfluencePage"),
                "page": 1
            }
            pinecone_vectors.append((vector_id, embedding,sparse_values, vector_metadata))
            local_counter += 1

            if (local_counter % batch_size == 0) or (idx == len(docs)):
                try:
                    log.info(
                        f"Upserting {len(pinecone_vectors)} vectors to Pinecone (batch upsert)."
                    )
                    index.upsert(vectors=pinecone_vectors, namespace=namespace)
                    log.info("Pinecone upsert successful.")

                    # **BEGIN: Update file status to 'processed' after successful upsert**
                    for vector in pinecone_vectors:
                        _, _, metadata = vector
                        page_id = metadata.get("pageId", "unknown_id")
                        file_id_to_update = generate_md5_hash(
                            sub_for_hash, project_id, data_source_id, page_id
                        )
                        try:
                            psql.update_file_status(
                                file_id=file_id_to_update,
                                status="processed",
                                page_count=1,  # Adjust if multiple pages per file
                                updated_at=datetime.now(timezone.utc),
                            )
                            log.info(f"Updated status to 'processed' for file_id: {file_id_to_update}")
                            used_credits = 1
                            message = json.dumps(
                                {
                                    "subscription_id": subscription_id,
                                    "data_source_id": data_source_id,
                                    "project_id": project_id,
                                    "creditsUsed": used_credits,
                                }
                            )
                            google_pub_sub.publish_messages_with_retry_settings(
                                GCP_PROJECT_ID, GCP_CREDIT_USAGE_TOPIC, message=message
                            )
                        except Exception as e:
                            log.error(f"Failed to update status for file_id {file_id_to_update}:")
                    # **END: Update file status to 'processed'**

                except Exception as e:
                    log.error("Error upserting to Pinecone:")
                finally:
                    pinecone_vectors.clear()

        log.info(
            f"Finished processing doc-chunks from pages: {batch_page_ids}"
        )

    log.info("Confluence ingestion (incremental sync) completed successfully.")


######################################################################
# MAIN ENTRYPOINT FOR CLOUD RUN JOB
######################################################################

def run_job():
    """
    Main entrypoint function for Cloud Run Job.
    Reads environment variables, parses data, then calls process_confluence_requests().
    """
    data_source_config = os.environ.get("DATA_SOURCE_CONFIG")
    if not data_source_config:
        log.error("DATA_SOURCE_CONFIG env var is missing.")
        sys.exit(1)

    try:
        config = json.loads(data_source_config)
        event_data = json.loads(config.get("event_data", "{}"))
        log.info(f"Cloud Run Job: parsed event_data: {event_data}")
    except Exception as e:
        log.error("Failed to parse DATA_SOURCE_CONFIG:")
        sys.exit(1)

    # Extract parameters
    confluence_url = event_data.get("confluenceUrl")
    confluence_user = event_data.get("confluenceUser")
    confluence_token = event_data.get("confluenceToken")
    parent_page_id = event_data.get("confluenceParent")
    last_sync_time = event_data.get("lastSyncTime")  # Possibly a string
    project_id = event_data.get("projectId")
    data_source_id = event_data.get("id")

    # Use projectId as Pinecone namespace, for example
    namespace = project_id

    # Retrieve subscriptionId from your DB if desired
    project_details = psql.get_project_details(project_id=project_id)
    subscription_id = project_details.get("subscriptionId") if project_details else None

    required_fields = {
        "confluenceURL": confluence_url,
        "confluenceUser": confluence_user,
        "confluenceToken": confluence_token,
        "parentPageId": parent_page_id,
        "projectId": project_id,
        "dataSourceId": data_source_id,
    }
    missing = [k for k, v in required_fields.items() if not v]
    if missing:
        log.error(f"Missing required fields: {missing}")
        sys.exit(1)

    try:
        # Mark data source as processing
        psql.update_data_source_by_id(data_source_id, status="processing")

        # Run ingestion
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

        # Mark data source as processed
        psql.update_data_source_by_id(data_source_id, status="processed")
        log.info("Job completed successfully.")
    except Exception as e:
        log.error("Error in process_confluence_requests:")
        psql.update_data_source_by_id(data_source_id, status="failed")
        sys.exit(1)


if __name__ == "__main__":
    # For local testing: just call run_job()
    run_job()
