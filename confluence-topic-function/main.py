import os
import sys
import json
import base64
import google.auth
import requests
from google.cloud import run_v2
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from shared import psql
from shared.logging_config import log

# Retrieve PostgreSQL connection parameters from environment variables
db_params = {
    "host": os.environ.get("PSQL_HOST"),
    "port": os.environ.get("PSQL_PORT", "5432"),  # Default port is 5432
    "database": os.environ.get("PSQL_DATABASE"),
    "user": os.environ.get("PSQL_USERNAME"),
    "password": os.environ.get("PSQL_PASSWORD"),
}

# Google Cloud Function entry point


def pubsub_to_cloud_run_confluence_job(event, context):
    """
    Trigger a Cloud Run Job using data received from a Pub/Sub topic.
    """
    # Decode the base64-encoded Pub/Sub message
    pubsub_message = event.get("data", "")
    if not pubsub_message:
        log.info("No data found in Pub/Sub message.")
        return

    try:
        message_data = base64.b64decode(pubsub_message).decode("utf-8")
        data = json.loads(message_data)
        log.info(f"Data received from topic: {data}")
    except Exception as e:
        log.error(f"Error decoding Pub/Sub message: {str(e)}")
        return

    # Extract required fields from the data
    dataSourceId = data.get("id")
    project_id = data.get("projectId")

    if not dataSourceId or not project_id:
        log.info("Missing required fields in the Pub/Sub event data.")
        return

    try:
        # Get data source status:
        data_source_details = psql.get_data_source_details(data_source_id=dataSourceId)
        if not data_source_details:
            log.info(f"No data source found with id {dataSourceId}")
            return

        status = data_source_details.get("status")

        if status in ["queued", None, ""]:
            # Cloud Run Job parameters
            # Job name from environment variable
            job_name = os.environ.get("CLOUD_RUN_JOB_NAME")
            # Job region from environment variable
            region = os.environ.get("CLOUD_RUN_REGION")
            # Project ID from environment variable
            project = os.environ.get("GCP_PROJECT_ID")
            # Create a Run client
            client = run_v2.JobsClient()
            job_path = client.job_path(project, region, job_name)
            job = client.get_job(request={"name": job_path})
            if not job_name or not region or not project:
                log.error("Missing environment variables for Cloud Run Job execution.")
                return

            # # Construct the REST API URL
            # run_job_url = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job_name}:run"

            # # Obtain an OAuth 2.0 access token using default credentials
            # credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            # credentials.refresh(google.auth.transport.requests.Request())
            # access_token = credentials.token

            # headers = {
            #     "Authorization": f"Bearer {access_token}",
            #     "Content-Type": "application/json",
            # }

            # # Prepare the request body with the event data
            # request_body = {
            #     "template": {
            #         "template": {
            #             "containers": [{
            #                 "args": [json.dumps(data)]  # Pass event data as argument
            #             }]
            #         }
            #     }
            # }
            event_data_str = json.dumps(data)

            log.info(f"Triggering Cloud Run Job: {job} with body: {event_data_str}")

            # Make the REST API call to trigger the Cloud Run Job
            # Construct the override
            struct_obj = Struct()
            struct_obj.update({"event_data": event_data_str})
            # Create a run object
            run_request = run_v2.RunJobRequest(
                name=job_name,
                overrides={
                    "container_overrides": [
                        {
                            "env": [
                                {
                                    "name": "DATA_SOURCE_CONFIG",
                                    "value": json_format.MessageToJson(struct_obj),
                                }
                            ]
                        }
                    ]
                },
            )

            # Execute the job and log
            response = client.run_job(request=run_request)
            print(f"Job execution started: {response.name}")

            if response.status_code in [200, 201]:
                log.info(
                    f"Cloud Run Job triggered successfully. Response: {response.json()}"
                )
            else:
                log.error(
                    f"Failed to trigger Cloud Run Job. Status Code: {response.status_code}, Response: {response.text}"
                )
                # Optionally, implement retry logic or publish to a dead-letter queue
    except Exception as e:
        log.error(f"Error triggering Cloud Run Job: {str(e)}")
        sys.exit("Function execution failed.")
