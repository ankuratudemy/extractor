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
$PROJECT_ID = "your-project-id"  # Replace with your actual project ID

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
        --ingress=internal=all
    $serviceURI = (gcloud run services describe $serviceName-$region --region $region --format 'value(status.url)')
    New-Variable -Name "$serviceName`_$region_URI" -Value $serviceURI -Force
}

# Create script block for deploying services
$scriptBlock = {
    param($serviceName, $region, $image, $cpu, $memory, $port, $max, $min, $customDomainAudience, $concurrency)
    Deploy-CloudRunService -serviceName $serviceName -region $region -image $image -cpu $cpu -memory $memory -port $port -max $max -min $min -customDomainAudience $customDomainAudience -concurrency $concurrency
}

# Create array to store job objects
$jobs = @()

# Iterate through each region and start a job for each service
foreach ($region in $REGIONS) {
    $feJob = Start-Job -ScriptBlock $scriptBlock -ArgumentList $FE_SERVICE_NAME_PREFIX, $region, $FE_IMAGE, $FE_CPU, $FE_MEMORY, $FE_PORT, $FE_MAX_INST, $FE_MIN_INST, $STRUCTHUB_DOMAIN_FE, $FE_CONCURRENT_REQUESTS_PER_INST
    $beJob = Start-Job -ScriptBlock $scriptBlock -ArgumentList $BE_SERVICE_NAME_PREFIX, $region, $BE_IMAGE, $BE_CPU, $BE_MEMORY, $BE_PORT, $BE_MAX_INST, $BE_MIN_INST, $STRUCTHUB_DOMAIN_BE, $BE_CONCURRENT_REQUESTS_PER_INST

    $jobs += $feJob, $beJob
}

# Wait for all jobs to complete
Wait-Job -Job $jobs | Out-Null

# Retrieve job results if needed
foreach ($job in $jobs) {
    $result = Receive-Job $job
    # Display or handle job results as needed
    Write-Host "Job result: $result"
}

# Remove jobs from the job history
Remove-Job -Job $jobs

# Continue with the rest of your script...
# Create global external IP addresses, network endpoint groups, backend services, URL maps, SSL certificates, target HTTPS proxies, and forwarding rules...

# Display the saved URLs for both services
foreach ($region in $REGIONS) {
    Write-Host "$FE_SERVICE_NAME_PREFIX`_$region_URI"
    Write-Host "$BE_SERVICE_NAME_PREFIX`_$region_URI"
}
