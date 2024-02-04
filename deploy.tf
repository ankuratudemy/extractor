provider "google" {
  credentials = file("path/to/your/credentials.json")
  project     = "your-project-id"
  region      = "us-central1" # Set your default region
}

variable "regions" {
  default = ["us-central1", "us-east4", "us-east1", "us-east5", "us-west1", "us-west2", "asia-south1", "asia-south2", "europe-west2", "europe-west3", "northamerica-northeast1", "northamerica-northeast2"]
}

variable "fe_max_inst" {
  default = "25"
}

variable "fe_min_inst" {
  default = "0"
}

variable "fe_cpu" {
  default = "1"
}

variable "fe_memory" {
  default = "2Gi"
}

variable "fe_port" {
  default = "5000"
}

variable "be_max_inst" {
  default = "75"
}

variable "be_min_inst" {
  default = "0"
}

variable "be_cpu" {
  default = "1"
}

variable "be_memory" {
  default = "2Gi"
}

variable "be_port" {
  default = "9998"
}

variable "external_ip_address_name_fe" {
  default = "xtract-fe-ip-name"
}

variable "external_ip_address_name_be" {
  default = "xtract-be-ip-name"
}

variable "structhub_domain_fe" {
  default = "stage.api.structhub.io"
}

variable "structhub_domain_be" {
  default = "stage-be.api.structhub.io"
}

variable "be_image" {
  default = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-be:1.0.0"
}

variable "fe_image" {
  default = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-fe:6.0.0"
}

variable "be_concurrent_requests_per_inst" {
  default = 1
}

variable "fe_concurrent_requests_per_inst" {
  default = 1
}

resource "google_project_service" "run" {
  service = "run.googleapis.com"
}

resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
}

# Iterate through each region and create Cloud Run service
resource "null_resource" "cloud_run_services" {
  count = length(var.regions)

  triggers = {
    region = var.regions[count.index]
  }

  provisioner "local-exec" {
    command = <<EOT
      gcloud run deploy xtract-fe-${var.regions[count.index]} \
        --region ${var.regions[count.index]} \
        --max-instances ${var.fe_max_inst} \
        --min-instances ${var.fe_min_inst} \
        --no-allow-unauthenticated \
        --image ${var.fe_image} \
        --set-env-vars SERVER_URL=${var.structhub_domain_be} \
        --cpu ${var.fe_cpu} \
        --memory ${var.fe_memory} \
        --port ${var.fe_port} \
        --add-custom-audiences ${var.structhub_domain_fe} \
        --concurrency ${var.fe_concurrent_requests_per_inst}

      gcloud run deploy xtract-be-${var.regions[count.index]} \
        --region ${var.regions[count.index]} \
        --max-instances ${var.be_max_inst} \
        --min-instances ${var.be_min_inst} \
        --no-allow-unauthenticated \
        --image ${var.be_image} \
        --set-env-vars SERVER_URL=${var.structhub_domain_be} \
        --cpu ${var.be_cpu} \
        --memory ${var.be_memory} \
        --port ${var.be_port} \
        --add-custom-audiences ${var.structhub_domain_be} \
        --concurrency ${var.be_concurrent_requests_per_inst}
    EOT
  }
}

# Create global external IP addresses for Cloud Run
resource "google_compute_address" "external_ip_fe" {
  name          = var.external_ip_address_name_fe
  network_tier  = "PREMIUM"
  ip_version    = "IPV4"
  global        = true
}

resource "google_compute_address" "external_ip_be" {
  name          = var.external_ip_address_name_be
  network_tier  = "PREMIUM"
  ip_version    = "IPV4"
  global        = true
}

# Iterate through each region and create Cloud Run Network Endpoint Group
resource "google_compute_network_endpoint_group" "neg_fe" {
  count                = length(var.regions)
  name                 = "neg-xtract-fe-${var.regions[count.index]}"
  region               = var.regions[count.index]
  network_endpoint_type = "serverless"
  cloud_run_service    = "xtract-fe-${var.regions[count.index]}"
}

resource "google_compute_network_endpoint_group" "neg_be" {
  count                = length(var.regions)
  name                 = "neg-xtract-be-${var.regions[count.index]}"
  region               = var.regions[count.index]
  network_endpoint_type = "serverless"
  cloud_run_service    = "xtract-be-${var.regions[count.index]}"
}

# Create global backend services for Cloud Run
resource "google_compute_backend_service" "fe_backend" {
  name                 = "xtract-fe-backend"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  global               = true
}

resource "google_compute_backend_service" "be_backend" {
  name                 = "xtract-be-backend"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  global               = true
}

# Iterate through each region and add Cloud Run Network Endpoint Group to global backend service
resource "google_compute_backend_service_backend" "fe_backend_backends" {
  count                        = length(var.regions)
  backend_service              = google_compute_backend_service.fe_backend.name
  network_endpoint_group      = google_compute_network_endpoint_group.neg_fe[count.index].name
  network_endpoint_group_region = var.regions[count.index]
}

resource "google_compute_backend_service_backend" "be_backend_backends" {
  count                        = length(var.regions)
  backend_service              = google_compute_backend_service.be_backend.name
  network_endpoint_group      = google_compute_network_endpoint_group.neg_be[count.index].name
  network_endpoint_group_region = var.regions[count.index]
}

# Create URL maps for Cloud Run services
resource "google_compute_url_map" "fe_url_map" {
  name          = "external-xtract-fe"
  default_service = google_compute_backend_service.fe_backend.name
}

resource "google_compute_url_map" "be_url_map" {
  name          = "external-xtract-be"
  default_service = google_compute_backend_service.be_backend.name
}

# Obtain external IP addresses for Cloud Run services
data "google_compute_address" "external_ip_fe_data" {
  name   = var.external_ip_address_name_fe
  global = true
}

data "google_compute_address" "external_ip_be_data" {
  name   = var.external_ip_address_name_be
  global = true
}

# Create SSL certificates for Cloud Run services
resource "google_compute_ssl_certificate" "fe_ssl_cert" {
  name        = "xtract-fe-structhub-cert"
  certificate = "path/to/your/certificate.pem" # Replace with actual certificate path
  private_key = "path/to/your/private-key.pem" # Replace with actual private key path
}

resource "google_compute_ssl_certificate" "be_ssl_cert" {
  name        = "xtract-be-structhub-cert"
  certificate = "path/to/your/certificate.pem" # Replace with actual certificate path
  private_key = "path/to/your/private-key.pem" # Replace with actual private key path
}

# Create target HTTPS proxies for Cloud Run services
resource "google_compute_target_https_proxy" "fe_https_proxy" {
  name                = "external-xtract-fe-https"
  ssl_certificates   = [google_compute_ssl_certificate.fe_ssl_cert.id]
  url_map            = google_compute_url_map.fe_url_map.id
}

resource "google_compute_target_https_proxy" "be_https_proxy" {
  name                = "external-xtract-be-https"
  ssl_certificates   = [google_compute_ssl_certificate.be_ssl_cert.id]
  url_map            = google_compute_url_map.be_url_map.id
}

# Create forwarding rules for Cloud Run services
resource "google_compute_global_forwarding_rule" "fe_forwarding_rule" {
  name                  = "xtract-fe-https"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  network_tier          = "PREMIUM"
  address               = google_compute_address.external_ip_fe_data.address
  target_https_proxy    = google_compute_target_https_proxy.fe_https_proxy.name
  port_range            = "443"
}

resource "google_compute_global_forwarding_rule" "be_forwarding_rule" {
  name                  = "xtract-be-https"
  load_balancing_scheme = "EXTERNAL_MANAGED"
  network_tier          = "PREMIUM"
  address               = google_compute_address.external_ip_be_data.address
  target_https_proxy    = google_compute_target_https_proxy.be_https_proxy.name
  port_range            = "443"
}

# Output the Cloud Run service URLs
output "service_urls" {
  value = [for region in var.regions : "xtract-fe-${region}-URI = <URL>, xtract-be-${region}-URI = <URL>"]
}
