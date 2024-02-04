$REGIONS = @("us-central1", "us-east4", "us-east1","us-east5", "us-west1","us-west2", "asia-south1","asia-south2","europe-west2","europe-west3","northamerica-northeast1","northamerica-northeast2")

$FE_MAX_INST="25"
$FE_MIN_INST="0"
$FE_CPU="1"
$FE_MEMORY="2Gi"
$FE_PORT="5000"
$BE_MAX_INST="75"
$BE_MIN_INST="0"
$BE_CPU="1"
$BE_MEMORY="2Gi"
$BE_PORT="9998"
$ExternalIpAddressNameFE = "xtract-fe-ip-name"
$ExternalIpAddressNameBE = "xtract-be-ip-name"
$STRUCTHUB_DOMAIN_FE="stage.api.structhub.io"
$STRUCTHUB_DOMAIN_BE="stage-be.api.structhub.io"
$BE_IMAGE="us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-be:1.0.0"
$FE_IMAGE="us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-fe:6.0.0"
$BE_CONCURRENT_REQUESTS_PER_INST=1
$FE_CONCURRENT_REQUESTS_PER_INST=1
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com compute.googleapis.com

# Set service names as variables
$FE_SERVICE_NAME_PREFIX = "xtract-fe"
$BE_SERVICE_NAME_PREFIX = "xtract-be"
$FE_HC_PATH = "/health"
$BE_HC_PATH = "/tika"


# Create global backend services for Cloud Run
gcloud compute backend-services update $FE_SERVICE_NAME_PREFIX-backend  --global --http-health-checks="http-health-check-$FE_SERVICE_NAME_PREFIX"
gcloud compute backend-services update $BE_SERVICE_NAME_PREFIX-backend  --global --http-health-checks="http-health-check-$BE_SERVICE_NAME_PREFIX"
