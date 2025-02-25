import os
import json
import logging
import datetime
from datetime import datetime
import mimetypes
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

# Google GenAI
from google import genai
from google.genai import types

# GCS
from google.cloud import storage

# PSQL Utility (adjust to your actual import & function)
from shared import psql

METADATA_FILEUPLOAD_BUCKET = os.environ.get("METADATA_FILEUPLOAD_BUCKET")

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

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
    # This pattern matches:
    # ```json
    #   (some stuff, possibly multiline)
    # ```
    # The (.*?) group captures everything inside the fences, non-greedy.
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


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
      8) Returns JSON or an empty object if still invalid
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
        metadata_keys = project_row.get("metadataKeys", None)
        if metadata_keys:
            metadata_keys = str(metadata_keys)
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
        gcs_blob_name = (
            f"metadata_files/{sub_id}/{timestamp_str}_{uploaded_file.filename}"
        )

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
        user_prompt = metadata_prompt + metadata_keys
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = google_genai_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[user_prompt, uri_part],
                    config={"temperature": 0.1},
                )
                # Possibly the model returns JSON wrapped in triple backticks
                response_text = extract_json_from_markdown(response.text)

                # Try JSON parse
                metadata_tags = json.loads(response_text)
                break  # success
            except json.JSONDecodeError:
                logging.warning(
                    f"Attempt {attempt+1} to parse JSON failed. Full response:\n{response.text}"
                )
                if attempt == max_attempts - 1:
                    # final attempt => log invalid
                    logging.error(
                        f"INVALID RESPONSE after {max_attempts} tries:\n{response.text}"
                    )
                    metadata_tags = {}
                else:
                    # optionally modify the prompt further if you wish
                    # e.g. user_prompt += "\nReturn strictly valid JSON with no markdown."
                    continue

        # Clean up local file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return jsonify({"metadataTags": metadata_tags}), 200

    except Exception as e:
        logging.error(f"GenAI or file upload error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=9999, debug=True)
