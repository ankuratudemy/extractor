import google.auth
from google.auth.transport.requests import AuthorizedSession
from google.auth.transport import requests
from google.oauth2 import service_account


def impersonated_id_token():
    credentials = service_account.Credentials.from_service_account_file(
        "./SAKeys/fe-key.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    # credentials, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    authed_session = AuthorizedSession(credentials)
    sa_to_impersonate = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    request_body = {"audience": "stage-be.api.structhub.io"}
    response = authed_session.post( f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa_to_impersonate}:generateIdToken',request_body)
    return response

if __name__ == "__main__":
    print(impersonated_id_token().json()['token'])