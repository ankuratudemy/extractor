from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account


def impersonated_id_token(serverurl):
    credentials = service_account.Credentials.from_service_account_file(
        "/app/key.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    # credentials, project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    authed_session = AuthorizedSession(credentials)
    sa_to_impersonate = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    request_body = {"audience": serverurl}
    response = authed_session.post( f'https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{sa_to_impersonate}:generateIdToken',request_body)
    return response