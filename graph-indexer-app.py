from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from neo4j import GraphDatabase
from langchain_google_vertexai.model_garden import ChatAnthropicVertex
import json
import tiktoken
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from concurrent.futures import ThreadPoolExecutor
import traceback

app = Flask(__name__)

# Configure Logging with JSON Formatter


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'level': record.levelname,
            'message': record.getMessage(),
            'timestamp': self.formatTime(record, self.datefmt),
            'name': record.name,
        }
        return json.dumps(log_record)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USER')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
MAX_CHUNK_TOKENS = int(os.environ.get('MAX_CHUNK_TOKENS', 100000))  # Default to 100000 if not set
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 10000))        # Default to 10000 if not set
# Validate Configuration
required_env_vars = ['NEO4J_URI', 'NEO4J_USER',
                     'NEO4J_PASSWORD', 'GCP_PROJECT_ID']
missing_env_vars = [
    var for var in required_env_vars if not os.environ.get(var)]
if missing_env_vars:
    logger.error(
        f"Missing environment variables: {', '.join(missing_env_vars)}")
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(missing_env_vars)}")

# Initialize Neo4j driver with neo4j+s URI
try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    logger.info("Connected to Neo4j successfully with encrypted connection.")
except Exception as e:
    logger.error("Failed to connect to Neo4j: " + str(e))
    logger.error(traceback.format_exc())
    raise e

# Initialize ChatAnthropicVertex
try:
    anthropic_chat = ChatAnthropicVertex(
        model_name="claude-3-5-sonnet@20240620",
        project=GCP_PROJECT_ID,
        location="us-east5",
        max_tokens=4096,
        temperature=0.5
    )
    logger.info("Initialized ChatAnthropicVertex successfully.")
except Exception as e:
    logger.error("Failed to initialize ChatAnthropicVertex: " + str(e))
    logger.error(traceback.format_exc())
    raise e

# Initialize Flask-Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'txt'}


def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_tokenizer():
    """Initialize and return the tokenizer."""
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info("Tokenizer initialized successfully.")
        return tokenizer
    except Exception as e:
        logger.error("Failed to initialize tokenizer: " + str(e))
        logger.error(traceback.format_exc())
        raise e


tokenizer = get_tokenizer()


def split_text_into_chunks(text, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
    """
    Split text into chunks by token count with specified overlap.
    This helps process large texts piece-by-piece.
    """
    logger.info("Starting text splitting into chunks.")
    try:
        tokens = tokenizer.encode(text)
        chunks = []
        start = 0
        text_length = len(tokens)

        while start < text_length:
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap
            if start < 0:
                start = 0

        logger.info(f"Split text into {len(chunks)} chunks successfully.")
        return chunks
    except Exception as e:
        logger.error("Error splitting text into chunks: " + str(e))
        logger.error(traceback.format_exc())
        raise e


PROMPT_DEDUPLICATION = """
You are an AI assistant specialized in maintaining data integrity within graph databases. Your task is to take a set of nodes and relationships, identify duplicates, and merge them to minimize fragmentation. The goal is to represent each unique real-world entity as a single node, even if multiple different names or properties appeared in the data.

**Given the following graph data:**

**Nodes:**
{NODES_JSON}

**Relationships:**
{RELATIONSHIPS_JSON}

**Instructions:**
1. Identify nodes referring to the same real-world entity. Merge them as much as possible.
2. Combine duplicates into a single node, consolidate properties.
3. Update relationships to reference merged nodes. Remove duplicates.
4. Maintain graph integrity and minimize fragmentation.
5. Return JSON with "nodes" and "relationships".

**Example Output:**
{
    "nodes": [
        {
            "id": "node1",
            "label": "Company",
            "properties": {
                "name": "Pacific Gas and Electric Company",
                "industry": "Utilities"
            }
        }
    ],
    "relationships": [
        {
            "start_id": "node1",
            "end_id": "node2",
            "type": "CERTIFIED",
            "properties": {
                "date": "2023-01-01"
            }
        }
    ]
}
Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
"""


@ app.route('/upload', methods=['POST'])
@ limiter.limit("10 per minute")
def upload_file():
    """
    Handle file uploads, process the text in chunks, extract graph data, deduplicate,
    and update Neo4j with a final consolidated, deduplicated graph.
    """
    logger.info("Received a file upload request.")
    if 'file' not in request.files:
        logger.warning("No file part in the request.")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        logger.warning("No file selected for uploading.")
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_dir = '/app/uploads'
        filepath = os.path.join(upload_dir, filename)
        try:
            os.makedirs(upload_dir, exist_ok=True)
            file.save(filepath)
            logger.info(f"Saved file to {filepath}.")
        except Exception as e:
            logger.error("Failed to save file: " + str(e))
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Failed to save file: {e}'}), 500

        # Process the file
        try:
            logger.info("Starting file processing.")
            process_file(filepath)
            logger.info("File processed successfully.")
            return jsonify({'message': 'File successfully processed'}), 200
        except Exception as e:
            logger.error("Error processing file: " + str(e))
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    else:
        logger.warning(f"File type not allowed: {file.filename}")
        return jsonify({'error': f'Allowed file types are {", ".join(ALLOWED_EXTENSIONS)}'}), 400


# ThreadPoolExecutor for concurrent processing of chunks
executor = ThreadPoolExecutor(max_workers=5)


def process_file(filepath):
    """
    Process the uploaded file:
    1. Read the entire text.
    2. Split into chunks.
    3. Extract nodes and relationships from each chunk concurrently.
    4. After all chunks processed, send all nodes and relationships to LLM for deduplication.
    5. Update Neo4j with final deduplicated graph.
    """
    logger.info(f"Reading file {filepath}.")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Read {len(text)} characters from {filepath}.")
    except Exception as e:
        logger.error("Failed to read file: " + str(e))
        logger.error(traceback.format_exc())
        raise e

    chunks = split_text_into_chunks(text)
    logger.info(f"Processing {len(chunks)} chunks concurrently.")

    all_nodes = []
    all_relationships = []

    # Process each chunk concurrently
    futures = []
    for idx, chunk_text in enumerate(chunks, start=1):
        futures.append(
            (executor.submit(extract_graph_from_text, chunk_text), idx, len(chunks)))

    # Collect results from each chunk
    for future, idx, total in futures:
        try:
            chunk_result = future.result()
            if chunk_result:
                chunk_nodes = chunk_result.get('nodes', [])
                chunk_relationships = chunk_result.get('relationships', [])
                all_nodes.extend(chunk_nodes)
                all_relationships.extend(chunk_relationships)
                logger.info(
                    f"Chunk {idx}/{total} processed: extracted {len(chunk_nodes)} nodes, {len(chunk_relationships)} relationships.")
            else:
                logger.warning(
                    f"No graph data extracted from chunk {idx}/{total}.")
        except Exception as e:
            logger.error(f"Error processing chunk {idx}/{total}: " + str(e))
            logger.error(traceback.format_exc())

    # Deduplicate nodes and relationships using the LLM
    logger.info(
        "Starting deduplication process for all nodes and relationships.")
    deduped_graph = deduplicate_graph_data(all_nodes, all_relationships)

    # Update Neo4j with deduplicated graph
    if deduped_graph:
        logger.info("Updating Neo4j with deduplicated graph data.")
        update_graphdb(deduped_graph)
    else:
        logger.warning("No deduplicated graph data returned from LLM.")


def extract_graph_from_text(text):
    """
    Extract nodes and relationships from a text chunk using the LLM.

    Expected JSON response:
    {
      "nodes": [...],
      "relationships": [...]
    }
    """
    logger.info("Extracting graph data from a text chunk using LLM.")
    extraction_prompt_template = """
You are an AI assistant that extracts structured graph data from unstructured text.

Given the following text, please extract all entities as nodes and the relationships between them. 
Return the result in a JSON format with two keys: "nodes" and "relationships".

"nodes": [
   {
     "id": "<unique_id>",
     "label": "<entity_type>",
     "properties": {
         "name": "<entity_name>"
     }
   }
],
"relationships": [
   {
     "start_id": "<node_id>",
     "end_id": "<node_id>",
     "type": "<relationship_type>",
     "properties": {
         "key": "value"
     }
   }
]

Ensure your JSON is well-formed and does not contain extra formatting.

**Text to process:**
{TEXT_TO_PROCESS}
Respond in JSON format. Do not add ```json at the beginning or ``` at the end. Do not duplicate sources in the `sources` array.
"""

    try:
        # Safely replace the placeholder with the actual text
        prompt = extraction_prompt_template.replace("{TEXT_TO_PROCESS}", text)
        # Debug-level to avoid cluttering logs
        logger.debug(f"Extraction Prompt: {prompt}")

        response = anthropic_chat.invoke(input=prompt)
        content = response.content.strip()
        logger.info("LLM extraction response received.")
        logger.debug(f"LLM Extraction Response: {content}")

        # Parse the response as JSON
        data = json.loads(content)
        nodes = data.get('nodes', [])
        relationships = data.get('relationships', [])
        logger.info(
            f"Extracted {len(nodes)} nodes and {len(relationships)} relationships from chunk.")
        return {'nodes': nodes, 'relationships': relationships}
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON response from LLM: " + str(e))
        logger.error(f"LLM Response Content: {content}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error("Error during LLM call for extraction: " + str(e))
        logger.error(traceback.format_exc())
        return None


def deduplicate_graph_data(nodes, relationships):
    """
    Send the aggregated nodes and relationships to the LLM for deduplication.
    Using .replace() to avoid invalid format specifiers and handle braces safely.
    """
    logger.info("Preparing deduplication prompt for LLM.")
    try:
        nodes_json = json.dumps(nodes, indent=4)
        relationships_json = json.dumps(relationships, indent=4)

        # Safely replace placeholders to avoid formatting issues
        prompt = PROMPT_DEDUPLICATION.replace("{NODES_JSON}", nodes_json).replace(
            "{RELATIONSHIPS_JSON}", relationships_json)
        # Debug-level to avoid cluttering logs
        logger.debug(f"Deduplication Prompt: {prompt}")

        logger.info("Invoking LLM for deduplication.")
        response = anthropic_chat.invoke(input=prompt)
        content = response.content.strip()
        logger.info("LLM Deduplication response received.")
        logger.debug(f"LLM Deduplication Response: {content}")

        # Parse and validate deduplicated data
        deduped_data = json.loads(content)
        deduped_nodes = deduped_data.get('nodes', [])
        deduped_relationships = deduped_data.get('relationships', [])

        if not isinstance(deduped_nodes, list) or not isinstance(deduped_relationships, list):
            logger.error(
                "Deduplication output must contain 'nodes' and 'relationships' as lists.")
            return None

        logger.info(
            f"After deduplication: {len(deduped_nodes)} nodes and {len(deduped_relationships)} relationships.")
        return {'nodes': deduped_nodes, 'relationships': deduped_relationships}
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON response from LLM: " + str(e))
        logger.error(f"LLM Deduplication Response Content: {content}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error("Error during LLM deduplication call: " + str(e))
        logger.error(traceback.format_exc())
        return None


def update_graphdb(graph_data):
    """
    Update the Neo4j graph with the deduplicated nodes and relationships.
    Uses MERGE to ensure idempotent updates.
    """
    if not graph_data:
        logger.warning("No graph data to update.")
        return

    nodes = graph_data.get('nodes', [])
    relationships = graph_data.get('relationships', [])
    logger.info(
        f"Updating Neo4j with {len(nodes)} nodes and {len(relationships)} relationships.")
    try:
        with driver.session() as session:
            session.write_transaction(create_nodes_batch, nodes)
            session.write_transaction(
                create_relationships_batch, relationships)
        logger.info("Deduplicated graph data updated successfully in Neo4j.")
    except Exception as e:
        logger.error("Failed to update deduplicated graph database: " + str(e))
        logger.error(traceback.format_exc())
        raise e


def create_nodes_batch(tx, nodes):
    """
    Create or update nodes in Neo4j in batch.
    """
    logger.info(f"Creating/Updating {len(nodes)} nodes in Neo4j.")
    for node in nodes:
        label = node.get('label', 'Node')
        tx.run(
            f"""
            MERGE (n:{label} {{id: $id}})
            SET n += $properties
            """,
            id=node['id'],
            properties=node.get('properties', {})
        )
    logger.info(f"{len(nodes)} nodes created/updated in batch.")


def create_relationships_batch(tx, relationships):
    """
    Create or update relationships in Neo4j in batch.
    """
    logger.info(
        f"Creating/Updating {len(relationships)} relationships in Neo4j.")
    for rel in relationships:
        rel_type = rel.get('type', 'RELATED_TO')
        tx.run(
            f"""
            MATCH (a {{id: $start_id}})
            WITH a
            MATCH (b {{id: $end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += $properties
            """,
            start_id=rel['start_id'],
            end_id=rel['end_id'],
            properties=rel.get('properties', {})
        )
    logger.info(
        f"{len(relationships)} relationships created/updated in batch.")


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    logger.info("Health check endpoint called.")
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    # Run the Flask app with debug=False for production
    logger.info("Starting Flask app.")
    app.run(host='0.0.0.0', port=5000, debug=False)
