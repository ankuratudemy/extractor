import os
import json
import logging
import datetime
from datetime import datetime
import mimetypes
import re
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# Google GenAI
from google import genai
from google.genai import types

# GCS
from google.cloud import storage

# PSQL Utility (adjust to your actual import & function)
from shared import psql, prompts

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


@app.route("/get_metadata", methods=["POST"])
def process_file():
    """
    This endpoint:
      1) Accepts an uploaded file (multipart/form-data)
      2) Uploads the file to GCS
      3) Dynamically detects MIME type (if possible)
      4) Calls Vertex Gemini with up to 3 attempts to parse JSON
         - If LLM returns JSON in markdown code fences, extract that first
      5) Returns JSON or an empty object if still invalid
    """
    # 1) Validate Input
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        # 5) Dynamically detect MIME type
        mime_type, _ = mimetypes.guess_type(uploaded_file.filename)

        # Read the file content
        file_content = uploaded_file.read()

        # Encode the file content in base64
        file_content_base64 = base64.b64encode(file_content)

        # Decode the base64-encoded content back to bytes
        file_content_bytes = base64.b64decode(file_content_base64)

        # Create the msg_document using the file content
        msg_document = types.Part.from_bytes(
            data=file_content_bytes,
            mime_type=mime_type,
        )

        # 6) Attempt up to 3 times to parse valid JSON
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
                    category="HARM_CATEGORY_HATE_SPEECH", threshold="ON"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="ON"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT", threshold="ON"
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
                return response.text
            except Exception as e:
                logging.warning(
                    f"Attempt {attempt+1} to get text from page failed. Error {str(e)}"
                )
                if attempt == max_attempts - 1:
                    # final attempt => log invalid
                    logging.error(
                        f"Text extraction failed after {max_attempts} tries:\n{str(e)}"
                    )
                    return None
                else:
                    continue
    except Exception as e:
        logging.warning(f"Error to get text from page failed. Error {str(e)}")
        return None


if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=9999, debug=True)
