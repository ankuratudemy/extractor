import json
import requests

def lambda_handler(event, context):
    print(event)
    # Extract data from the Lambda event
    data = json.loads(event["body"])
    content = data.get("content", "")

    # Make a POST request to the Tika server
    tika_url = "http://127.0.0.1:5000/tika"
    headers = {'Accept': 'text/plain', 'Content-Type': 'application/pdf'}
    response = requests.post(tika_url, data=content, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        tika_result = response.text
        return {
            "statusCode": 200,
            "body": json.dumps({"tika_result": tika_result}),
        }
    else:
        return {
            "statusCode": response.status_code,
            "body": json.dumps({"error": "Failed to process content"}),
        }
