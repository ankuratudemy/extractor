import os
import json
import logging
import mimetypes
import re
import base64
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# Google GenAI
from google import genai
from google.genai import types

# GCS
from google.cloud import storage

# PSQL Utility (adjust to your actual import & function)
from shared import psql, prompts

# For robust date/datetime parsing:
from dateutil.parser import parse as dateutil_parse, ParserError

METADATA_FILEUPLOAD_BUCKET = os.environ.get("METADATA_FILEUPLOAD_BUCKET")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Global GenAI client
google_genai_client = genai.Client(
    vertexai=True,
    project=os.environ.get("GCP_PROJECT_ID"),  # or your project ID
    location="us-central1",
)

def extract_json_from_markdown(text: str) -> str:
    """
    If the LLM output has ```json ...``` code fences, extract the content within.
    Otherwise, return the original text.
    """
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def convert_dates_for_metadata_keys(metadata_dict: dict, metadata_keys: list) -> dict:
    """
    Given the dictionary returned by the LLM (metadata_dict) and the list of metadata_keys from DB,
    only convert values to a Unix timestamp if:
      - The key is found in metadata_dict,
      - The corresponding metadata_keys entry has type 'Date', and
      - The value in metadata_dict is parseable as a date/datetime.

    If the conversion fails for any reason, the original value is left intact.
    This function operates on top-level keys only.
    """
    # Build a lookup: { 'someKey': 'Date'/'String'/'List'/'Number', ... }
    db_key_map = {}
    for mk in metadata_keys:
        k = mk.get('key')
        t = mk.get('type')
        if k and t:
            db_key_map[k] = t

    for k, v in metadata_dict.items():
        if k in db_key_map and db_key_map[k] == "Date":
            if isinstance(v, str):
                try:
                    dt = dateutil_parse(v)
                    metadata_dict[k] = int(dt.timestamp())
                except (ParserError, ValueError):
                    # If conversion fails, leave the original string
                    pass
    return metadata_dict

@app.route("/get_metadata", methods=["POST"])
def process_file():
    """
    This endpoint:
      1) Accepts an uploaded file (multipart/form-data)
      2) Requires 'project_id' and 'sub_id' in form data
      3) Looks up metadataPrompt, metadataKeys from PSQL
      4) If missing, returns {metadataTags:{}}
      5) Uploads the file to GCS
      6) Dynamically detects MIME type (if possible)
      7) Calls Vertex Gemini with up to 3 attempts to parse JSON
         - If LLM returns JSON in markdown code fences, extract that first
      8) Once metadata_tags is created, if a key is typed as 'Date' in DB,
         try to parse it with dateutil and replace that field with a Unix timestamp.
         If conversion fails, the original date string is returned.
      9) Returns final JSON.
    """
    # 1) Validate Input
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    project_id = request.form.get("project_id")
    sub_id = request.form.get("sub_id")
    if not project_id or not sub_id:
        return jsonify({"error": "Missing project_id or sub_id"}), 400

    # 2) Fetch Project Details from PSQL
    try:
        project_row = psql.get_project_details(project_id)
        if not project_row:
            return jsonify({"error": f"No project found with id={project_id}"}), 404

        metadata_prompt: str = project_row.get("metadataPrompt", None)

        # metadataKeys from DB might be a JSON string or list.
        metadata_keys_raw = project_row.get("metadataKeys", None)
        if metadata_keys_raw:
            if isinstance(metadata_keys_raw, str):
                try:
                    metadata_keys = json.loads(metadata_keys_raw)
                except Exception:
                    metadata_keys = []
            elif isinstance(metadata_keys_raw, list):
                metadata_keys = metadata_keys_raw
            else:
                metadata_keys = []
        else:
            metadata_keys = []
    except Exception as e:
        logging.error(f"Database error: {e}")
        return jsonify({"error": str(e)}), 500

    # 3) Check if we have prompt & keys
    if not metadata_prompt or not metadata_keys:
        return jsonify({"metadataTags": {}}), 200

    try:
        # 4) Save to local temp, then upload to GCS
        tmp_path = os.path.join("/tmp", uploaded_file.filename)
        uploaded_file.save(tmp_path)

        timestamp_str = datetime.utcnow().isoformat().replace(":", "-")
        gcs_blob_name = f"metadata_files/{sub_id}/{timestamp_str}_{uploaded_file.filename}"

        storage_client = storage.Client()
        bucket = storage_client.bucket(METADATA_FILEUPLOAD_BUCKET)
        blob = bucket.blob(gcs_blob_name)
        blob.upload_from_filename(tmp_path)

        gcs_uri = f"gs://{METADATA_FILEUPLOAD_BUCKET}/{gcs_blob_name}"
        logging.info(f"Uploaded file to {gcs_uri}")

        # 5) Dynamically detect MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.filename)
        if mime_type:
            uri_part = types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
        else:
            uri_part = types.Part.from_uri(file_uri=gcs_uri)

        # 6) Attempt up to 3 times to parse valid JSON
        metadata_tags = {}
        user_prompt = metadata_prompt + str(metadata_keys)
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = google_genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[user_prompt, uri_part],
                    config={"temperature": 0.1},
                )
                response_text = extract_json_from_markdown(response.text)
                logging.info(f"Metadata Tags Generated (attempt {attempt+1}):\n {response_text}")
                metadata_tags = json.loads(response_text)
                break  # success, exit the loop
            except json.JSONDecodeError:
                logging.warning(
                    f"Attempt {attempt+1} to parse JSON failed. Full response:\n{response.text}"
                )
                if attempt == max_attempts - 1:
                    logging.error(
                        f"INVALID RESPONSE after {max_attempts} tries:\n{response.text}"
                    )
                    metadata_tags = {}
                else:
                    continue

        # 7) Convert date fields to Unix timestamps for keys defined as 'Date'
        metadata_tags = convert_dates_for_metadata_keys(metadata_tags, metadata_keys)

        # Clean up local file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return jsonify({"metadataTags": metadata_tags}), 200

    except Exception as e:
        logging.error(f"GenAI or file upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/get_text", methods=["POST"])
def get_text():
    """
    This endpoint:
      1) Accepts an uploaded file (multipart/form-data)
      2) Uploads the file to GCS
      3) Dynamically detects MIME type (if possible)
      4) Calls Vertex Gemini with up to 3 attempts to extract text
         - If LLM returns JSON in markdown code fences, extract that first
      5) Returns text or an empty object if still invalid
    """
    # 1) Validate Input
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        mime_type, _ = mimetypes.guess_type(uploaded_file.filename)
        file_content = uploaded_file.read()
        file_content_base64 = base64.b64encode(file_content)
        file_content_bytes = base64.b64decode(file_content_base64)

        msg_document = types.Part.from_bytes(
            data=file_content_bytes,
            mime_type=mime_type,
        )

        SI = prompts.TEXT_EXTRACT_PROMPT
        user_prompt = "Extract text from attached file"
        max_attempts = 3
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"
                ),
            ],
            system_instruction=[types.Part.from_text(text=SI)],
        )
        for attempt in range(max_attempts):
            try:
                response = google_genai_client.models.generate_content(
                    model="gemini-2.0-flash-lite-001",
                    contents=[user_prompt, msg_document],
                    config=generate_content_config,
                )
                logging.info(f"Response from gemini {response.text}")
                return response.text
            except Exception as e:
                logging.error(
                    f"Attempt {attempt+1} to get text from page failed. Error {str(e)}"
                )
                if attempt == max_attempts - 1:
                    logging.error(
                        f"Text extraction failed after {max_attempts} tries:\n{str(e)}"
                    )
                    return None
                else:
                    continue
    except Exception as e:
        logging.error(f"Error to get text from page failed. Error {str(e)}")
        return None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
