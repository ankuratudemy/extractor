from flask import Flask, jsonify, request
from google.cloud import storage, exceptions
import os, time
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    matching_engine_index_config,
)

app = Flask(__name__)


@app.route('/create_project', methods=['POST'])
def create_project():
    # Get project name from request
    data = request.get_json()
    if not data or 'projectName' not in data:
        return jsonify({'error': 'Missing projectName in request'}), 400
    if not data or 'username' not in data:
        return jsonify({'error': 'Missing username in request'}), 400
    username = data['username']
    project_name = data["projectName"]

    # Check if bucket already exists
    storage_client = storage.Client()
    bucket_name = f"{project_name}-bucket-{username}"
    try:
        storage_client.get_bucket(bucket_name)
        print(f"Bucket '{bucket_name}' already exists. Using existing bucket.")
    except exceptions.NotFound:
        # Create bucket if it doesn't exist
        bucket = storage_client.create_bucket(bucket_or_name=bucket_name, location="us-central1")
        print(f"Bucket '{bucket_name}' created.")

    bucket_uri = f"gs://{bucket_name}"

    # Check if Vertex Search Index exists
    try:
        existing_indexes = aiplatform.MatchingEngineIndex.list(location="us-central1")
        filtered_index = None
        for index in existing_indexes:
            if index.display_name == project_name:
                filtered_index = index
                break
        print(f"Vertex Search Index '{project_name}' already exists. Using existing index. with id {filtered_index.name}")
        if not filtered_index:
            # Create index if it doesn't exist
            index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=project_name,
                contents_delta_uri=bucket_uri,
                shard_size="SHARD_SIZE_SMALL",
                description="Structhub project index",
                dimensions=768,  # Adjust dimensions based on your data
                approximate_neighbors_count=150,
                index_update_method="STREAM_UPDATE",  # Adjust as needed
                distance_measure_type=matching_engine_index_config.DistanceMeasureType.DOT_PRODUCT_DISTANCE.value
            )
            print(f"Vertex Search Index '{project_name}' created.")
    except Exception as e:
        # Handle other potential exceptions (e.g., permission issues)
        return jsonify({'error': f"Error creating Vertex Search Index: {str(e)}"}), 500

    # Check if Vertex Search Endpoint exists
    try:
        existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(location="us-central1")
        filtered_endpoint = None
        for endpoint in existing_endpoints:
            if endpoint.display_name == f"{project_name}-endpoint":
                filtered_endpoint = endpoint
                break
        print(f"Vertex Search Index Endpoint '{project_name}-endpoint' already exists. Using existing endpoint with id {filtered_endpoint.name}")
        # Create endpoint if it doesn't exist
        if not endpoint:
            endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=f"{project_name}-endpoint", public_endpoint_enabled=True, location="us-central1"
            )
            print(f"Vertex Search Endpoint '{project_name}-endpoint' created.")
    except Exception as e:
        # Handle other potential exceptions (e.g., permission issues)
        return jsonify({'error': f"Error creating Vertex Search Endpoint: {str(e)}"}), 500

    # Deploy index to endpoint (if necessary)
    try:
        if endpoint.deployed_indexes or len(endpoint.deployed_indexes) < 1:
            print(f"Deployed Index Id {index.display_name.replace('-', '_')}")
            endpoint.deploy_index(index=index, deployed_index_id=index.display_name.replace('-', '_'))
            print(f"Vertex Search Index '{index.name}' deployed to endpoint '{endpoint.name}'.")
    except Exception as e:
        return jsonify({'error': f"Error deploying index to endpoint: {str(e)}"}), 500

    # Return success message
    return jsonify({
        'message': 'Project created successfully!',
        'bucketName': bucket_name,
        'vertexSearchEndpoint': endpoint.name,
        'vertexSearchName' : index.name
    }) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)