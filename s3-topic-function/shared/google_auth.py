from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
import time

MAX_RETRIES = 5

def impersonated_id_token(serverurl):
    credentials = service_account.Credentials.from_service_account_file(
        "/app/key.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    sa_to_impersonate = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    request_body = {"audience": serverurl}

    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Create a new AuthorizedSession for each attempt
            authed_session = AuthorizedSession(credentials)
            response = authed_session.post(f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa_to_impersonate}:generateIdToken', json=request_body)

            # Check the response status code and handle retries accordingly
            if response.status_code == 200:
                return response
            else:
                print(f"Error in POST request. Retrying... Status code: {response.status_code}")
                retries += 1
                time.sleep(1)  # You may adjust the sleep duration between retries

        except Exception as e:
            print(f"Error during POST request: {str(e)}")
            retries += 1
            time.sleep(1)  # You may adjust the sleep duration between retries

    # If retries are exhausted, raise an exception or handle it as needed
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries in impersonated_id_token")
