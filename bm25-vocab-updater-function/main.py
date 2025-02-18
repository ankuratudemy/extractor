# bm25_vocab_updater.py
import os
import base64
import json
import logging
from math import log

from google.cloud import firestore
from google.cloud.firestore import Increment
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firestore client with 'database' parameter
db = firestore.Client(
    project=os.getenv("GCP_PROJECT_ID"),
    database=os.getenv("FIRESTORE_DB")
)

@firestore.transactional
def update_top_level(transaction, doc_ref, doc_length):
    """
    Transactional function to update the top-level BM25 vocabulary document.

    Args:
        transaction (firestore.Transaction): The active Firestore transaction.
        doc_ref (firestore.DocumentReference): Reference to the top-level BM25 document.
        doc_length (int): The length of the document being added.
    """
    try:
        # Retrieve the current snapshot of the document within the transaction
        snaps = transaction.get(doc_ref)
        snap = next(snaps)
        if snap.exists:
            data = snap.to_dict()
            logger.debug(f"Existing data for project_id='{doc_ref.id}': {data}")
        else:
            # Initialize fields if the document does not exist
            data = {"N": 0, "total_doc_length": 0, "avgdl": 0.0}
            logger.debug(f"Initializing data for new project_id='{doc_ref.id}': {data}")

        # Update the fields
        data["N"] += 1
        data["total_doc_length"] += doc_length
        data["avgdl"] = data["total_doc_length"] / data["N"] if data["N"] > 0 else 0.0

        logger.debug(f"Updated data for project_id='{doc_ref.id}': {data}")

        # Set the updated data back to Firestore within the transaction
        transaction.set(doc_ref, data)

    except Exception as e:
        logger.exception(f"Transaction failed for project_id='{doc_ref.id}': {str(e)}")
        raise  # Re-raise the exception to trigger a transaction retry if applicable

def pubsub_to_bm25_vocab_updater(event, context):
    """
    Cloud Function entry point: merges partial BM25 updates into Firestore.
    Expects a Pub/Sub message with a JSON payload like:
      {
        "project_id": "...",
        "doc_term_freq": { "term1": freq1, "term2": freq2, ... },
        "doc_length": int,
        "is_new_doc": bool
      }
    """
    try:
        # Check if the incoming event has "data"
        if "data" not in event:
            logger.error("No data in event. Exiting.")
            return

        # Decode the Pub/Sub message
        message_data = base64.b64decode(event["data"]).decode("utf-8")
        partial_update = json.loads(message_data)
        logger.info(f"Received partial update: {partial_update}")

        # Extract fields from the partial update
        project_id = partial_update["project_id"]
        doc_term_freq = partial_update["doc_term_freq"]
        doc_length = partial_update["doc_length"]
        is_new_doc = partial_update.get("is_new_doc", True)
        transaction = db.transaction()
        # 1) Update the top-level document if it's a new doc
        if is_new_doc:
            # Reference to the top-level doc in the "bm25Vocabs" collection
            top_level_ref = db.collection("bm25Vocabs").document(project_id)

            # Call the transactional function without passing the transaction
            update_top_level(transaction, top_level_ref, doc_length)

        # 5) Batch update the "terms" subcollection in chunks to avoid 500-write limit
        terms_items = list(doc_term_freq.items())
        CHUNK_SIZE = 400  # Safe upper bound below the 500 limit

        for i in range(0, len(terms_items), CHUNK_SIZE):
            chunk = terms_items[i : i + CHUNK_SIZE]
            batch = db.batch()

            for term, freq in chunk:
                term_doc_ref = (
                    db.collection("bm25Vocabs")
                    .document(project_id)
                    .collection("terms")
                    .document(term)
                )

                snap = term_doc_ref.get()
                if snap.exists:
                    data = snap.to_dict() or {"df": 0, "tf": 0}
                else:
                    data = {"df": 0, "tf": 0}

                # If it's a new document, increment document frequency once
                if is_new_doc:
                    data["df"] = data.get("df", 0) + 1

                # Always add to the total term frequency
                data["tf"] = data.get("tf", 0) + freq

                batch.set(term_doc_ref, data)

            batch.commit()

            logger.debug(f"Committed a batch of {len(chunk)} term updates.")

        logger.info(f"Successfully updated BM25 subcollection for project_id='{project_id}'.")

    except Exception as e:
        logger.exception(f"Error processing BM25 vocab update: {str(e)}")
