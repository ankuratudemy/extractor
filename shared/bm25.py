"""
shared/bm25.py

Removes SpaCy usage entirely. If you need tokenization, we show a simple regex-based
approach. Otherwise, you can skip it.
"""

import os
import re
import json
import math
import hashlib
from google.cloud import pubsub_v1, firestore
from .logging_config import log
from .google_pub_sub import publish_messages_with_retry_settings

BM25_VOCAB_UPDATES_TOPIC = os.getenv("BM25_VOCAB_UPDATES_TOPIC")
FIRESTORE_DB = os.getenv("FIRESTORE_DB")
db = firestore.Client(project=os.getenv("GCP_PROJECT_ID"), database=FIRESTORE_DB)
publisher = pubsub_v1.PublisherClient() if BM25_VOCAB_UPDATES_TOPIC else None

# Default BM25 parameters
K1 = 1.5
B  = 0.75


###############################################################################
# 1) Simple Regex Tokenization
###############################################################################
def tokenize_document(text: str):
    """
    Example naive tokenizer using regex. Splits on non-alphanumeric characters,
    lowercases, and filters out empty tokens. Does NOT remove stopwords, etc.
    Adjust as needed.
    """
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return tokens

###############################################################################
# 2) Publish Partial BM25 Updates
###############################################################################
def publish_partial_bm25_update(project_id: str, tokens: list, is_new_doc: bool = True):
    """
    Publishes an incremental BM25 update to BM25_VOCAB_UPDATES_TOPIC for downstream merges.
    doc_term_freq, doc_length, is_new_doc are included in the message.
    """
    if not BM25_VOCAB_UPDATES_TOPIC or not publisher:
        log.warning("BM25_VOCAB_UPDATES_TOPIC not set. Skipping BM25 update publish.")
        return

    doc_term_freq = {}
    for t in tokens:
        doc_term_freq[t] = doc_term_freq.get(t, 0) + 1

    partial_update = {
        "project_id": project_id,
        "doc_term_freq": doc_term_freq,
        "doc_length": len(tokens),
        "is_new_doc": is_new_doc
    }

    message_data = json.dumps(partial_update)

    try:
        publish_messages_with_retry_settings(os.getenv("GCP_PROJECT_ID"),BM25_VOCAB_UPDATES_TOPIC,message_data)
        log.info(f"Published BM25 partial update for project_id={project_id} to {BM25_VOCAB_UPDATES_TOPIC}")
    except Exception as e:
        log.exception(f"Failed to publish partial BM25 update: {e}")

###############################################################################
# 3) Compute BM25-based Sparse Vector
###############################################################################
def compute_bm25_sparse_vector(
    tokens: list,
    project_id: str,
    vocab_stats: dict,
    max_terms: int = 300
):
    """
    Computes a BM25-based sparse vector for the given tokens and project vocab stats.
    Returns a dict: {
        "indices": [...],
        "values": [...]
    }
    suitable for Pinecone's "sparseValues".

    :param tokens: A list of tokens from the doc.
    :param project_id: The project/vocab ID, if needed for references.
    :param vocab_stats: A dictionary containing:
        {
          "N": int,  # total doc count
          "avgdl": float,
          "vocab": { term -> { df, tf } } # optional or partial
        }
    :param max_terms: The maximum number of terms to include in the final sparse vector
                      (sort by descending weight, limit to top-K).
    """
    N     = vocab_stats.get("N", 1)     # avoid zero
    avgdl = vocab_stats.get("avgdl", 1) # avoid zero
    doc_length = len(tokens)

    def get_df(term):
        if "vocab" in vocab_stats:
            tinfo = vocab_stats["vocab"].get(term)
            if tinfo:
                return tinfo.get("df", 0)
        return 0

    local_freq = {}
    for t in tokens:
        local_freq[t] = local_freq.get(t, 0) + 1

    # BM25
    K1_CONST = 1.5
    B_CONST  = 0.75

    term_weights = []
    for term, freq in local_freq.items():
        df = get_df(term)
        numerator   = freq * (K1_CONST + 1)
        denominator = freq + K1_CONST*(1 - B_CONST + B_CONST*(doc_length / avgdl))
        # IDF
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        bm25_score  = idf * (numerator / denominator)
        if bm25_score > 0:
            term_weights.append((term, bm25_score))

    # Sort descending
    term_weights.sort(key=lambda x: x[1], reverse=True)
    top_terms = term_weights[:max_terms]

    indices, values = [], []
    for (term, weight) in top_terms:
        dim = stable_term_hash(term)
        indices.append(dim)
        values.append(weight)

    return {"indices": indices, "values": values}

def stable_term_hash(term: str, modulo: int = 100_000_000):
    """
    Simple stable hashing of term => dimension. 
    In production, store a "term->dimension" mapping to avoid collisions.
    """
    md5val = hashlib.md5(term.encode("utf-8")).hexdigest()
    numeric_val = int(md5val, 16)
    return numeric_val % modulo

###############################################################################
# 4) (Optional) Upsert Sparse Vector to Pinecone
###############################################################################
def upsert_sparse_vector_to_pinecone(
    doc_id: str,
    project_id: str,
    sparse_vector: dict,
    metadata: dict = None,
    pinecone_index = None
):
    """
    Upserts a sparse vector to Pinecone using "sparseValues". 
    If you also have a dense embedding, you can pass it in "values" field.

    :param doc_id: The unique ID for this doc in Pinecone
    :param project_id: Used as Pinecone namespace
    :param sparse_vector: A dict { "indices": [...], "values": [...] }
    :param metadata: Additional Pinecone metadata
    :param pinecone_index: A Pinecone index client instance
    """
    if pinecone_index is None:
        log.error("No Pinecone index client provided. Skipping upsert.")
        return

    if not sparse_vector.get("indices") or not sparse_vector.get("values"):
        log.warning(f"Empty sparse vector for doc_id={doc_id}. Skipping upsert.")
        return

    vector_data = {
        "id": doc_id,
        "sparseValues": {
            "indices": sparse_vector["indices"],
            "values": sparse_vector["values"],
        }
    }
    if metadata:
        vector_data["metadata"] = metadata

    try:
        pinecone_index.upsert(
            vectors=[vector_data],
            namespace=project_id
        )
        log.info(f"Upserted sparse vector to Pinecone for doc_id={doc_id} in namespace={project_id}")
    except Exception as e:
        log.exception(f"Failed to upsert sparse vector to Pinecone: {e}")


###############################################################################
# 5) Example: Loading vocab stats from Firestore
###############################################################################
def get_project_vocab_stats(project_id: str) -> dict:
    """
    Fetches BM25 stats for the given projectId from Firestore.
    Expects a doc: bm25Vocabs/{projectId} with fields: {N, total_doc_length, avgdl}
    And a subcollection "terms" with docs: term -> {df, tf}
    
    If no doc is found, returns default stats.
    """
    doc_ref = db.collection("bm25Vocabs").document(project_id)
    doc_snap = doc_ref.get()

    if not doc_snap.exists:
        log.warning(f"No BM25 vocab doc for project_id={project_id}. Returning default stats.")
        return {
            "N": 1,
            "avgdl": 1,
            "vocab": {}
        }

    top_data = doc_snap.to_dict()
    N     = top_data.get("N", 1)
    avgdl = top_data.get("avgdl", 1)

    vocab_data = {}
    terms_collection = doc_ref.collection("terms")
    for tdoc_snap in terms_collection.stream():
        tdata = tdoc_snap.to_dict()
        vocab_data[tdoc_snap.id] = {
            "df": tdata.get("df", 0),
            "tf": tdata.get("tf", 0)
        }

    return {
        "N": N,
        "avgdl": avgdl,
        "vocab": vocab_data
    }
