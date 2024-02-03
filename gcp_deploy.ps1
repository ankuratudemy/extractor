$REGIONS = @("us-central1", "us-east4")

$FE_MAX_INST="25"
$FE_MIN_INST="0"
$FE_CPU="2"
$FE_MEMORY="4Gi"
$FE_PORT="5000"
$BE_MAX_INST="50"
$BE_MIN_INST="0"
$BE_CPU="1"
$BE_MEMORY="2Gi"
$BE_PORT="9998"
$ExternalIpAddressNameFE = "xtract-fe-ip-name"
$ExternalIpAddressNameBE = "xtract-be-ip-name"
$STRUCTHUB_DOMAIN_FE="stage.api.structhub.io"
$STRUCTHUB_DOMAIN_BE="stage-be.api.structhub.io"
$BE_IMAGE="us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-be:1.0.0"
$FE_IMAGE="us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-fe:5.0.0"
$BE_CONCURRENT_REQUESTS_PER_INST=1
$FE_CONCURRENT_REQUESTS_PER_INST=1
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com compute.googleapis.com

# Set service names as variables
$FE_SERVICE_NAME_PREFIX = "xtract-fe"
$BE_SERVICE_NAME_PREFIX = "xtract-be"

# Function to deploy Cloud Run service
Function Deploy-CloudRunService {
    param(
        [string]$serviceName,
        [string]$region,
        [string]$image,
        [string]$cpu,
        [string]$memory,
        [string]$port,
        [string]$max,
        [string]$min,
        [string]$customDomainAudience,
        [string]$concurrency
    )

    gcloud run deploy $serviceName-$region `
        --region $region `
        --max-instances $max `
        --min-instances $min `
        --no-allow-unauthenticated `
        --image $image `
        --set-env-vars SERVER_URL=$STRUCTHUB_DOMAIN_BE `
        --cpu $cpu `
        --memory $memory `
        --port $port `
        --add-custom-audiences $customDomainAudience `
        --concurrency $concurrency 
    $serviceURI = (gcloud run services describe $serviceName-$region --region $region --format 'value(status.url)')
    New-Variable -Name "$serviceName`_$region_URI" -Value $serviceURI -Force
}

# Deploy xtract-fe and xtract-be services in each region
foreach ($region in $REGIONS) {
    Deploy-CloudRunService -serviceName $FE_SERVICE_NAME_PREFIX -region $region -image $FE_IMAGE -cpu $FE_CPU -memory $FE_MEMORY -port $FE_PORT -max $FE_MAX_INST -min $FE_MIN_INST -customDomainAudience $STRUCTHUB_DOMAIN_FE -concurrency $FE_CONCURRENT_REQUESTS_PER_INST
    Deploy-CloudRunService -serviceName $BE_SERVICE_NAME_PREFIX -region $region -image $BE_IMAGE -cpu $BE_CPU -memory $BE_MEMORY -port $BE_PORT -max $BE_MAX_INST -min $BE_MIN_INST -customDomainAudience $STRUCTHUB_DOMAIN_BE -concurrency $BE_CONCURRENT_REQUESTS_PER_INST
}

# Create global external IP addresses for Cloud Run
gcloud compute addresses create $ExternalIpAddressNameFE --network-tier=PREMIUM --ip-version=IPV4 --global
gcloud compute addresses create $ExternalIpAddressNameBE --network-tier=PREMIUM --ip-version=IPV4 --global

# Iterate through each region and create Cloud Run Network Endpoint Group
foreach ($region in $REGIONS) {
    $networkEndpointGroupNameFE = "neg-$FE_SERVICE_NAME_PREFIX-$region"
    $networkEndpointGroupNameBE = "neg-$BE_SERVICE_NAME_PREFIX-$region"

    gcloud compute network-endpoint-groups create $networkEndpointGroupNameFE --region $region --network-endpoint-type=serverless --cloud-run-service=$FE_SERVICE_NAME_PREFIX-$region
    gcloud compute network-endpoint-groups create $networkEndpointGroupNameBE --region $region --network-endpoint-type=serverless --cloud-run-service=$BE_SERVICE_NAME_PREFIX-$region
}

# Create global backend services for Cloud Run
gcloud compute backend-services create $FE_SERVICE_NAME_PREFIX-backend --load-balancing-scheme=EXTERNAL_MANAGED --global
gcloud compute backend-services create $BE_SERVICE_NAME_PREFIX-backend --load-balancing-scheme=EXTERNAL_MANAGED --global

# Iterate through each region and add Cloud Run Network Endpoint Group to global backend service
foreach ($region in $REGIONS) {
    $networkEndpointGroupNameFE = "neg-$FE_SERVICE_NAME_PREFIX-$region"
    $networkEndpointGroupNameBE = "neg-$BE_SERVICE_NAME_PREFIX-$region"

    gcloud compute backend-services add-backend $FE_SERVICE_NAME_PREFIX-backend --global --network-endpoint-group=$networkEndpointGroupNameFE --network-endpoint-group-region=$region
    gcloud compute backend-services add-backend $BE_SERVICE_NAME_PREFIX-backend --global --network-endpoint-group=$networkEndpointGroupNameBE --network-endpoint-group-region=$region
}

# Create URL maps for Cloud Run services
gcloud compute url-maps create external-$FE_SERVICE_NAME_PREFIX --default-service $FE_SERVICE_NAME_PREFIX-backend
gcloud compute url-maps create external-$BE_SERVICE_NAME_PREFIX --default-service $BE_SERVICE_NAME_PREFIX-backend

# Obtain external IP addresses for Cloud Run services
$EXTERNAL_IP_FE = (gcloud compute addresses describe $ExternalIpAddressNameFE --format="get(address)" --global)
$EXTERNAL_IP_BE = (gcloud compute addresses describe $ExternalIpAddressNameBE --format="get(address)" --global)

# Create SSL certificates for Cloud Run services
gcloud compute ssl-certificates create $FE_SERVICE_NAME_PREFIX-structhub-cert --domains "$STRUCTHUB_DOMAIN_FE"
gcloud compute ssl-certificates create $BE_SERVICE_NAME_PREFIX-structhub-cert --domains "$STRUCTHUB_DOMAIN_BE"

# Create target HTTPS proxies for Cloud Run services
gcloud compute target-https-proxies create external-$FE_SERVICE_NAME_PREFIX-https --ssl-certificates=$FE_SERVICE_NAME_PREFIX-structhub-cert --url-map=external-$FE_SERVICE_NAME_PREFIX
gcloud compute target-https-proxies create external-$BE_SERVICE_NAME_PREFIX-https --ssl-certificates=$BE_SERVICE_NAME_PREFIX-structhub-cert --url-map=external-$BE_SERVICE_NAME_PREFIX

# Create forwarding rules for Cloud Run services
gcloud compute forwarding-rules create $FE_SERVICE_NAME_PREFIX-https --load-balancing-scheme=EXTERNAL_MANAGED --network-tier=PREMIUM --address=$ExternalIpAddressNameFE --target-https-proxy=external-$FE_SERVICE_NAME_PREFIX-https --global --ports=443
gcloud compute forwarding-rules create $BE_SERVICE_NAME_PREFIX-https --load-balancing-scheme=EXTERNAL_MANAGED --network-tier=PREMIUM --address=$ExternalIpAddressNameBE --target-https-proxy=external-$BE_SERVICE_NAME_PREFIX-https --global --ports=443

# Display the saved URLs for both services
foreach ($region in $REGIONS) {
    Write-Host "$FE_SERVICE_NAME_PREFIX`_$region_URI"
    Write-Host "$BE_SERVICE_NAME_PREFIX`_$region_URI"
}

