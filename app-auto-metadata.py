#!/usr/bin/env python3
# app-metadata-indexer.py

import os
import io
import json
import sys
import signal
import logging
from datetime import datetime

from flask import Flask, request
from google.cloud import storage
from google import genai
from google.genai import types

# psql references your shared psql.py (with update_project, get_project_details, etc.)
from shared import psql
from shared.common_code import (
    generate_md5_hash,
    shutdown_handler,
    CENTRAL_TZ,
    UPLOADS_FOLDER,
)

app = Flask(__name__)
app.debug = True

# Environment variables
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")

# Initialize a global GenAI client
google_genai_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location="us-central1",
)

# Our special prompt that instructs the model to produce at least 20 keys in the JSON.
# You can store it directly here, or pull from environment, or from DB, etc.
METADATA_PROMPT = """[  {   "key": "<keyname with lowercase with no special characters and '_' for spaces>",   "type": "List or Date",   "Description": "Describe the key here and how to pick this value from document"  }  ][  {   "key": "<keyname with lowercase with no special characters and '_' for spaces>",   "type": "List/Date/String",   "Description": "Describe the key here and how to pick this value from document. Be as descriptibve as possible about this."  }  ]
for example##
[
  {
    "key": "filing_type",
    "type": "List",
    "Description": "This is type of financial filing the document is for. example 10-k,10-q, 8-k, proxy statement, form 3 etc. Always create at least 5 variation of filing name. example for 10-k, create values like [\"10k\",\"10-k\",\"10 k\", and so on] . "
  },
  {
    "key": "company_name",
    "type": "List",
    "Description": "Name of the company the data is about. Create multiple variable of company names with jumbled words, abbreviations, full forms etc. Create at least 10 variations. Make sure you pick a company or organisation's name as value. For example: Adobe, Microsoft, Google are names of companies."
  }
]

You are an AI agent tasked to understand the document provided as highly expert in the topic or domain document is about. Understand the document like a professional and generate at least 20 keys with dtype and description of key as mentioned 
in above format. Provide only JSON formatted list of keys and nothing else. Nothing before and after the JSON.
ALWAYS use type as `List` or `Date`.
As shown in example you are not supposed to pick the actual values or facts for description for this specific document but to create a general description which tells what the key is about and how to get this value from simialr documents
"""


def extract_json_from_markdown(text: str) -> str:
    """
    If the LLM output has ```json ...``` code fences, extract the content within.
    Otherwise, return the original text.
    """
    import re

    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


@app.route("/", methods=["POST"])
def event_handler():
    """
    Triggered by a GCS event (Cloud Storage -> Pub/Sub -> Cloud Run).
    1) Parse JSON event; extract bucket, filename (plus subscription, project, user from path).
    2) Download file from GCS to local /tmp or UPLOADS_FOLDER.
    3) Invoke Gemini with METADATA_PROMPT, attach the file as a 'Part'.
    4) Parse the model's JSON.
    5) Update the Project table with the returned JSON, setting it in 'metadataKeys' column.
    """
    project_id = None
    try:
        event_data = request.get_json()
        if not event_data:
            return ("No event data", 400)

        bucket_name = event_data.get("bucket")
        file_name = event_data.get("name")
        if not bucket_name or not file_name:
            return ("Missing bucket or file name in event data", 400)

        # Extract subscription_id, project_id, user_id from path
        folder_name, file_name_only = os.path.split(file_name)
        folder_name_parts = folder_name.split("/")
        if len(folder_name_parts) < 3:
            logging.error("Invalid folder path format for GCS object.")
            return ("Invalid folder path format", 400)

        subscription_id = folder_name_parts[0]
        project_id = folder_name_parts[1]
        user_id = folder_name_parts[2]
        logging.info(
            f"Subscription={subscription_id}, Project={project_id}, User={user_id}"
        )

        # Download the file from GCS
        local_temp_path = os.path.join(UPLOADS_FOLDER, file_name_only)
        logging.info(f"Local file temp path: {local_temp_path}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.download_to_filename(local_temp_path)
        logging.info(f"Downloaded {file_name} to {local_temp_path}")

        # Step: Call Gemini model with the content
        # Read file bytes
        with open(local_temp_path, "rb") as f:
            file_bytes = f.read()

        # Attempt to guess a MIME type (optional)
        import mimetypes

        mime_type, _ = mimetypes.guess_type(file_name_only)

        # Create a Part from bytes
        part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

        # We'll do multiple attempts to parse valid JSON.
        metadata_keys_json = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # We provide the prompt + the file as separate "contents"
                response = google_genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[METADATA_PROMPT, part],
                    config={"temperature": 0.1, "max_output_tokens": 2048},
                )
                logging.info(f"LLM response (attempt {attempt+1}): {response.text}")

                # Extract potential JSON from code fences
                response_text_clean = extract_json_from_markdown(response.text)

                # Try to load it
                metadata_keys_json = json.loads(response_text_clean)
                break  # success, break out of loop
            except json.JSONDecodeError:
                logging.warning(f"JSON parse failed on attempt {attempt+1}.")
                if attempt == max_attempts - 1:
                    logging.error(
                        "Unable to parse valid JSON from model after multiple attempts."
                    )
                    metadata_keys_json = []  # or None
            except Exception as e:
                logging.error(f"Error calling Gemini or parsing JSON: {e}")
                if attempt == max_attempts - 1:
                    metadata_keys_json = []

        # Cleanup local file
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)

        # If we ended up with no valid JSON, just store an empty list
        if not metadata_keys_json:
            metadata_keys_json = []

        # (Optional) we might also store the prompt used.
        # For example, you could do: psql.update_project(project_id, metadata_prompt=METADATA_PROMPT, metadata_keys=metadata_keys_json)
        # Or if you only want to store the keys, pass None for prompt:
        DEFAULT_METADATA_PROMPT = "From the uploaded document, think like an expert in it's domain and provided list  `metadata_keys` below with key name , description on how to pick the values for the key, and type of output like List, String value, or a number, you are to provide a JSON output with final key values generated for all keys in metadata_keys list. If no value if found just put `None` as value. Make sure you keep all values in lowercase. key names should be same as provided in metadata_keys list. Always create atleast 10 variations of values where type is `List` by creating jumbled word variation, synonyms, abbreviations, full forms, etc"
        psql.update_project(
            project_id=project_id,
            metadata_prompt=DEFAULT_METADATA_PROMPT,  # or None
            metadata_keys=metadata_keys_json,
        )

        return (f"Metadata keys updated for project={project_id}", 200)

    except Exception as e:
        logging.exception(f"Exception in event_handler: {str(e)}")
        return (f"Exception in event_handler: {e}", 500)


if __name__ == "__main__":
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER, exist_ok=True)

    signal.signal(signal.SIGINT, shutdown_handler)
    app.run(host="0.0.0.0", port=8080, debug=True)
else:
    signal.signal(signal.SIGTERM, shutdown_handler)
