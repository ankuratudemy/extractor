provider "google" {
  # credentials = file("/dev/null")
  project = "structhub-412620"
  region  = "us-central1"
}


variable "environment" {
  description = "Environment: 'stage'"
  type        = string
  default     = "stage"
}

locals {
  environment                          = var.environment # Set the desired environment here
  us_regions                           = ["us-central1"]
  regions                              = var.environment == "prod" ? ["northamerica-northeast1", "northamerica-northeast2", "us-central1", "us-east4", "us-east1", "us-west1", "us-west2", "us-west3", "us-west4", "us-south1", "asia-south1", "asia-south2", "europe-west1", "europe-west2", "europe-west3", "europe-west4", "australia-southeast1", "asia-southeast1", "asia-east1"] : ["northamerica-northeast1", "northamerica-northeast2", "us-central1", "us-east4", "us-east1", "us-west1", "us-west2", "us-west3", "us-west4", "us-south1", "asia-south1", "asia-south2", "europe-west1", "europe-west2", "europe-west3", "europe-west4", "australia-southeast1", "asia-southeast1", "asia-east1"]
  fe_cpu                               = 2
  fe_memory                            = "2Gi"
  fe_port                              = 5000
  be_cpu                               = 1
  be_memory                            = "2Gi"
  be_port                              = 9998
  xlsx_cpu                             = 1
  xlsx_memory                          = "2Gi"
  xlsx_port                            = 9999
  metadata_cpu                         = 1
  metadata_memory                      = "2Gi"
  metadata_port                        = 9997
  indexer_cpu                          = 1
  indexer_memory                       = "2Gi"
  indexer_port                         = 5000
  metadata_keys_cpu                    = 1
  metadata_keys_memory                 = "2Gi"
  metadata_keys_port                  = 5000
  searxng_port                         = 8080
  confluence_cpu                       = 1
  confluence_memory                    = "2Gi"
  confluence_port                      = 5000
  gdrive_cpu                           = 1
  gdrive_memory                        = "2Gi"
  gdrive_port                          = 5000
  azureblob_cpu                        = 1
  azureblob_memory                     = "2Gi"
  azureblob_port                       = 5000
  s3_cpu                               = 1
  s3_memory                            = "2Gi"
  s3_port                              = 5000
  gcpbucket_cpu                        = 1
  gcpbucket_memory                     = "2Gi"
  gcpbucket_port                       = 5000
  onedrive_cpu                         = 1
  onedrive_memory                      = "2Gi"
  onedrive_port                        = 5000
  sharepoint_cpu                       = 1
  sharepoint_memory                    = "2Gi"
  sharepoint_port                      = 5000
  external_ip_address_name_fe          = "xtract-fe-ip-name"
  internal_ip_address_name_indexer     = "xtract-indexer-ip-name"
  external_ip_address_name_be          = "xtract-be-ip-name"
  be_image                             = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-be:17.0.0"
  xlsx_image                           = "us-central1-docker.pkg.dev/structhub-412620/xtract/xlsx-indexer:16.0.0"
  metadata_image                       = "us-central1-docker.pkg.dev/structhub-412620/xtract/metadata:14.0.0"
  fe_image                             = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-fe:gcr-288.0.0"
  indexer_image                        = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-indexer:89.0.0"
  metadata_keys_image                  = "us-central1-docker.pkg.dev/structhub-412620/xtract/metadata-keys:8.0.0"
  websearch_image                      = "us-central1-docker.pkg.dev/structhub-412620/xtract/searxng:6.0.0"
  gdrive_image                         = "us-central1-docker.pkg.dev/structhub-412620/xtract/googledrive-indexer:42.0.0"
  confluence_image                     = "us-central1-docker.pkg.dev/structhub-412620/xtract/confluence-indexer-30.0.0"
  onedrive_image                       = "us-central1-docker.pkg.dev/structhub-412620/xtract/onedrive-indexer:23.0.0"
  sharepoint_image                     = "us-central1-docker.pkg.dev/structhub-412620/xtract/sharepoint-indexer:24.0.0"
  s3_image                             = "us-central1-docker.pkg.dev/structhub-412620/xtract/s3-indexer:32.0.0"
  azureblob_image                      = "us-central1-docker.pkg.dev/structhub-412620/xtract/azureblob-indexer:26.0.0"
  gcpbucket_image                      = "us-central1-docker.pkg.dev/structhub-412620/xtract/gcpbucket-indexer:32.0.0"
  be_concurrent_requests_per_inst      = 1
  fe_concurrent_requests_per_inst      = 1
  indexer_concurrent_requests_per_inst = 1
  project_id                           = "structhub-412620"
  project_number                       = "485124114765"
  fe_service_name_prefix               = "xtract-fe"
  indexer_service_name_prefix          = "xtract-indexer"
  metadata_keys_service_name_prefix    = "metadata-keys"
  be_service_name_prefix               = "xtract-be"
  xlsx_service_name_prefix             = "xlsx-be"
  metadata_service_name_prefix         = "metadata-be"
  fe_hc_path                           = "/health"
  be_hc_path                           = "/tika"
  fe_domain_suffix                     = local.environment == "prod" ? "" : "-stage"
  indexer_domain_suffix                = local.environment == "prod" ? "" : "-stage"
  metadata_keys_domain_suffix          = local.environment == "prod" ? "" : "-stage"
  be_domain_suffix                     = local.environment == "prod" ? "" : "-stage"
  websearch_domain_suffix              = local.environment == "prod" ? "" : "-stage"
  xlsx_domain_suffix                   = local.environment == "prod" ? "" : "-stage"
  metadata_domain_suffix               = local.environment == "prod" ? "" : "-stage"

  region_instance_counts = {
    "northamerica-northeast1" = {
      fe_max_inst      = local.environment == "prod" ? 1000 : 1000
      fe_min_inst      = 0
      indexer_max_inst = local.environment == "prod" ? 1000 : 1000
      indexer_min_inst = 0
      be_max_inst      = local.environment == "prod" ? 1000 : 1000
      be_min_inst      = 0
    }
    "northamerica-northeast2" = {
      fe_max_inst      = local.environment == "prod" ? 500 : 500
      fe_min_inst      = 0
      be_max_inst      = local.environment == "prod" ? 500 : 500
      be_min_inst      = 0
    }
    "us-central1" = {
      fe_max_inst      = 12000
      fe_min_inst      = 0
      indexer_max_inst = 12000
      indexer_min_inst = 0
      be_max_inst      = 12000
      be_min_inst      = 0
    }
    "us-south1" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "us-east4" = {
      fe_max_inst      = 5000
      fe_min_inst      = 0
      indexer_max_inst = 5000
      indexer_min_inst = 0
      be_max_inst      = 5000
      be_min_inst      = 0
    }
    "us-east1" = {
      fe_max_inst      = 7000
      fe_min_inst      = 0
      indexer_max_inst = 7000
      indexer_min_inst = 0
      be_max_inst      = 7000
      be_min_inst      = 0
    }
    "us-east5" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "us-west1" = {
      fe_max_inst      = 4500
      fe_min_inst      = 0
      indexer_max_inst = 4500
      indexer_min_inst = 0
      be_max_inst      = 4500
      be_min_inst      = 0
    }
    "us-west2" = {
      fe_max_inst      = 4500
      fe_min_inst      = 0
      indexer_max_inst = 4500
      indexer_min_inst = 0
      be_max_inst      = 4500
      be_min_inst      = 0
    }
    "us-west3" = {
      fe_max_inst      = 3000
      fe_min_inst      = 0
      indexer_max_inst = 3000
      indexer_min_inst = 0
      be_max_inst      = 3000
      be_min_inst      = 0
    }
    "us-west4" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "asia-south1" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "asia-south2" = {
      fe_max_inst      = 1000
      fe_min_inst      = 0
      indexer_max_inst = 1000
      indexer_min_inst = 0
      be_max_inst      = 1000
      be_min_inst      = 0
    }
    "asia-southeast1" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "asia-east1" = {
      fe_max_inst      = 4000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 4000
      be_min_inst      = 0
    }
    "asia-south2" = {
      fe_max_inst      = 1000
      fe_min_inst      = 0
      indexer_max_inst = 4000
      indexer_min_inst = 0
      be_max_inst      = 1000
      be_min_inst      = 0
    }
    "europe-west1" = {
      fe_max_inst      = 9000
      fe_min_inst      = 0
      indexer_max_inst = 9000
      indexer_min_inst = 0
      be_max_inst      = 9000
      be_min_inst      = 0
    }
    "europe-west2" = {
      fe_max_inst      = 5000
      fe_min_inst      = 0
      indexer_max_inst = 5000
      indexer_min_inst = 0
      be_max_inst      = 5000
      be_min_inst      = 0
    }
    "europe-west3" = {
      fe_max_inst      = 4500
      fe_min_inst      = 0
      indexer_max_inst = 4500
      indexer_min_inst = 0
      be_max_inst      = 4500
      be_min_inst      = 0
    }
    "europe-west4" = {
      fe_max_inst      = 5000
      fe_min_inst      = 0
      indexer_max_inst = 5000
      indexer_min_inst = 0
      be_max_inst      = 5000
      be_min_inst      = 0
    }
    "australia-southeast1" = {
      fe_max_inst      = 5000
      fe_min_inst      = 0
      indexer_max_inst = 5000
      indexer_min_inst = 0
      be_max_inst      = 5000
      be_min_inst      = 0
    }
  }
}


resource "google_project_service" "compute_api" {
  service                    = "compute.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "drive_api" {
  service                    = "drive.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

# Enable Eventarc API
resource "google_project_service" "eventarc" {
  service            = "eventarc.googleapis.com"
  disable_on_destroy = false
}

# Enable Pub/Sub API
resource "google_project_service" "pubsub" {
  service            = "pubsub.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "run_api" {
  service                    = "run.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "storage_api" {
  service                    = "storage.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "firebase_api" {
  project                    = local.project_id
  service                    = "firebase.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}


# resource "google_compute_address" "external_ip_fe" {
#   name = local.external_ip_address_name_fe
# }

# resource "google_compute_address" "external_ip_be" {
#   name = local.external_ip_address_name_be
# }

resource "google_firestore_database" "firestore" {
  name        = "structhub-${local.environment}"
  location_id = "us-central1"
  type        = "FIRESTORE_NATIVE"
  project     = local.project_id
  # Ensure the Firebase API is enabled before creating the Firebase project
  depends_on = [
    google_project_service.firebase_api
  ]
}


resource "google_compute_global_address" "external_ip_fe" {
  name = "${local.external_ip_address_name_fe}${local.fe_domain_suffix}"
}

resource "google_compute_global_address" "external_ip_be" {
  name = "${local.external_ip_address_name_be}${local.be_domain_suffix}"
}

resource "google_compute_backend_service" "fe_backend" {
  name                            = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-backend"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  connection_draining_timeout_sec = 310
  locality_lb_policy              = "RANDOM"
  enable_cdn                      = false

  dynamic "backend" {
    for_each = local.regions

    content {
      group = google_compute_region_network_endpoint_group.fe_backend[backend.key].id
    }
  }

  depends_on = [
    google_project_service.compute_api,
  ]
}

resource "google_compute_backend_service" "be_backend" {
  name                            = "${local.be_service_name_prefix}${local.be_domain_suffix}-backend"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  connection_draining_timeout_sec = 185
  locality_lb_policy              = "RANDOM"
  enable_cdn                      = false

  dynamic "backend" {
    for_each = local.regions

    content {
      group = google_compute_region_network_endpoint_group.be_backend[backend.key].id
    }
  }

  depends_on = [
    google_project_service.compute_api,
  ]
}

resource "google_compute_managed_ssl_certificate" "fe_ssl_cert" {
  name = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-structhub-cert"
  managed {
    domains = [local.environment == "prod" ? "api.structhub.io" : "stage.api.structhub.io"]
  }
}

resource "google_compute_managed_ssl_certificate" "be_ssl_cert" {
  name = "${local.be_service_name_prefix}${local.be_domain_suffix}-structhub-cert"
  managed {
    domains = [local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"]
  }
}

resource "google_compute_target_https_proxy" "fe_https_proxy" {
  name                        = "external-${local.fe_service_name_prefix}${local.fe_domain_suffix}-https"
  ssl_certificates            = [google_compute_managed_ssl_certificate.fe_ssl_cert.id]
  url_map                     = google_compute_url_map.fe_url_map.id
  http_keep_alive_timeout_sec = 610

  depends_on = [
    google_compute_managed_ssl_certificate.fe_ssl_cert
  ]
}

resource "google_compute_target_https_proxy" "be_https_proxy" {
  name                        = "external-${local.be_service_name_prefix}${local.be_domain_suffix}-https"
  ssl_certificates            = [google_compute_managed_ssl_certificate.be_ssl_cert.id]
  url_map                     = google_compute_url_map.be_url_map.id
  http_keep_alive_timeout_sec = 610

  depends_on = [
    google_compute_managed_ssl_certificate.be_ssl_cert
  ]
}

resource "google_compute_url_map" "fe_url_map" {
  name            = "external-${local.fe_service_name_prefix}${local.fe_domain_suffix}"
  default_service = google_compute_backend_service.fe_backend.id
}

resource "google_compute_url_map" "be_url_map" {
  name            = "external-${local.be_service_name_prefix}${local.be_domain_suffix}"
  default_service = google_compute_backend_service.be_backend.id
}

resource "google_compute_global_forwarding_rule" "fe_forwarding_rule" {
  name                  = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-https"
  target                = google_compute_target_https_proxy.fe_https_proxy.id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.external_ip_fe.id
  port_range            = "443"
  depends_on            = [google_compute_target_https_proxy.fe_https_proxy]
}

resource "google_compute_global_forwarding_rule" "be_forwarding_rule" {
  name                  = "${local.be_service_name_prefix}${local.be_domain_suffix}-https"
  target                = google_compute_target_https_proxy.be_https_proxy.id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.external_ip_be.id
  port_range            = "443"
  depends_on            = [google_compute_target_https_proxy.be_https_proxy]
}

resource "google_compute_region_network_endpoint_group" "fe_backend" {
  count                 = length(local.regions)
  name                  = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.regions[count.index]
  cloud_run {
    service = google_cloud_run_v2_service.fe_cloud_run[local.regions[count.index]].name
  }
}

resource "google_compute_region_network_endpoint_group" "be_backend" {
  count                 = length(local.regions)
  name                  = "${local.be_service_name_prefix}${local.be_domain_suffix}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.regions[count.index]
  cloud_run {
    service = google_cloud_run_v2_service.be_cloud_run[local.regions[count.index]].name
  }
}

resource "google_compute_region_network_endpoint_group" "xlsx_backend" {
  count                 = length(local.regions)
  name                  = "${local.xlsx_service_name_prefix}${local.xlsx_domain_suffix}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.regions[count.index]
  cloud_run {
    service = google_cloud_run_v2_service.xlsx_cloud_run[local.regions[count.index]].name
  }
}
resource "google_cloud_run_v2_service" "fe_cloud_run" {
  for_each = toset(local.regions)
  name     = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-${each.key}"
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    scaling {
      max_instance_count = local.region_instance_counts[each.key].fe_max_inst
      min_instance_count = local.region_instance_counts[each.key].fe_min_inst
    }
    containers {
      ports {
        container_port = local.fe_port
      }
      image = local.fe_image
      startup_probe {
        initial_delay_seconds = 0
        timeout_seconds       = 1
        period_seconds        = 3
        failure_threshold     = 1
        tcp_socket {
          port = local.fe_port
        }
      }
      env {
        name  = "SERVER_URL"
        value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
      }
      env {
        name  = "WEBSEARCH_SERVER_URL"
        value = local.environment == "prod" ? "websearch.structhub.io" : "stage-websearch.structhub.io"
      }
      env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
      }
      env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
      }
      env {
        name  = "GCP_CREDIT_USAGE_TOPIC"
        value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
      }
      env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
      env {
        name  = "UPLOADS_FOLDER"
        value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
      }
      env {
        name = "REDIS_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
            version = "latest"
          }
        }
      }

      env {
        name = "PSQL_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_USERNAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_DATABASE"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "PINECONE_INDEX_NAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "GROQ_API_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "GROQ_API_KEY" : "GROQ_API_KEY_STAGE"
            version = "latest"
          }
        }
      }
      resources {
        limits = {
          cpu    = local.fe_cpu
          memory = local.fe_memory
        }
      }
    }
    timeout                          = "690s"
    max_instance_request_concurrency = local.fe_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  depends_on = [
    google_project_service.run_api
  ]
}

resource "google_cloud_run_service_iam_binding" "fe_cloud_run_iam_binding" {
  for_each = toset(local.regions)

  location = each.key
  service  = google_cloud_run_v2_service.fe_cloud_run[each.key].name
  role     = "roles/run.invoker"
  members = [
    "allUsers",
  ]
}

resource "google_cloud_run_service_iam_binding" "be_cloud_run_iam_binding" {
  for_each = toset(local.regions)

  location = each.key
  service  = google_cloud_run_v2_service.be_cloud_run[each.key].name
  role     = "roles/run.invoker"
  members = [
    "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com",
  ]
}

resource "google_cloud_run_v2_service" "be_cloud_run" {
  for_each = toset(local.regions)

  name     = "${local.be_service_name_prefix}${local.be_domain_suffix}-${each.key}"
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    scaling {
      max_instance_count = local.region_instance_counts[each.key].be_max_inst
      min_instance_count = local.region_instance_counts[each.key].be_min_inst
    }
    containers {
      ports {
        container_port = local.be_port
      }
      image = local.be_image
      resources {
        limits = {
          cpu    = local.be_cpu
          memory = local.be_memory
        }
      }
    }
    timeout                          = "180s"
    max_instance_request_concurrency = local.be_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  custom_audiences = [local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"]
  depends_on = [
    google_project_service.run_api
  ]
}

resource "google_pubsub_topic" "credit_usage_dead_letter_topic" {
  name                       = "structhub-credit-usage-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}

resource "google_pubsub_topic" "credit_usage_topic" {
  name                       = "structhub-credit-usage-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}

# Generates an archive of the source code compressed as a .zip file.
data "archive_file" "credit_usage_function_source" {
  type        = "zip"
  source_dir  = "../../credit-usage-function"
  output_path = "${path.module}/function.zip"
}

# Add source code zip to the Cloud Function's bucket (Cloud_function_bucket) 
resource "google_storage_bucket_object" "zip" {
  source       = "${path.module}/function.zip"
  content_type = "application/zip"
  name         = "credit-usage-function${local.fe_domain_suffix}-${data.archive_file.credit_usage_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.function_bucket.name
  depends_on = [
    google_storage_bucket.function_bucket,
    data.archive_file.credit_usage_function_source
  ]
}

resource "google_cloudfunctions2_function" "credit_usage_function" {
  name     = "credit-usage-function${local.fe_domain_suffix}"
  location = "us-central1"
  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
    pubsub_topic   = google_pubsub_topic.credit_usage_topic.id
  }
  build_config {
    source {
      storage_source {
        bucket = google_storage_bucket.function_bucket.name
        object = "credit-usage-function${local.fe_domain_suffix}-${data.archive_file.credit_usage_function_source.output_md5}.zip"
      }
    }
    entry_point = "pubsub_to_postgresql"
    runtime     = "python310"
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true
    environment_variables = {
      MY_ENV_VAR = "sample_env_for_future_ref"
    }

    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "REDIS_HOST"
      secret     = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "REDIS_PASSWORD"
      secret     = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "REDIS_PORT"
      secret     = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
      version    = "latest"
    }


  }
  depends_on = [
    google_pubsub_topic.credit_usage_topic,
    google_pubsub_topic.credit_usage_dead_letter_topic, # Ensure the dead letter topic is created first
  ]
}

resource "google_storage_bucket" "function_bucket" {
  name     = "credit-usage-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}


# Create a subscription so that we don't lose messages published to dead letter topic
resource "google_pubsub_subscription" "credit_usage_dead_letter_subscription" {
  name  = "credit_usage_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.credit_usage_dead_letter_topic.name
}

# Grant permission to receive Eventarc events
resource "google_project_iam_member" "indexer_eventreceiver" {
  project = local.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}
# Grant permission to receive Eventarc events
resource "google_project_iam_member" "indexer_aiplatform" {
  project = local.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}
# Grant Pub/Sub Publisher role to the GCS service account
resource "google_project_iam_member" "gcs_pubsubpublisher" {
  project = local.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:service-${local.project_number}@gs-project-accounts.iam.gserviceaccount.com"
}
# Grant permission to invoke Cloud Run services
resource "google_project_iam_member" "indexer_runinvoker" {
  project = local.project_id
  role    = "roles/run.invoker"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}

resource "google_pubsub_topic" "fileupload_event_topic" {
  name = "file-upload-event-topic-${local.environment}"
}

resource "google_storage_bucket" "fileupload_bucket" {
  name                        = "structhub-file-upload-bucket-${local.environment}"
  location                    = "us"
  uniform_bucket_level_access = true
  cors {
    origin          = ["https://stage.structhub.io", "http://localhost:3000", "https://structhub.io"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}


resource "google_project_iam_member" "indexer_pubsubpublisher" {
  project = local.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}
resource "google_project_iam_member" "eventreceiver" {
  project = local.project_id
  role    = "roles/eventarc.eventReceiver"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}
resource "google_project_iam_member" "secretaccessor" {
  project = local.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
}

resource "google_cloud_run_v2_service" "indexer_cloud_run" {
  for_each = toset(local.us_regions)

  name     = "${local.indexer_service_name_prefix}${local.indexer_domain_suffix}-${each.key}"
  location = each.key
  template {
    service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    scaling {
      max_instance_count = local.region_instance_counts[each.key].indexer_max_inst
      min_instance_count = local.region_instance_counts[each.key].indexer_min_inst
    }

    containers {
      ports {
        container_port = local.indexer_port
      }
      image = local.indexer_image
      env {
        name  = "SERVER_URL"
        value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
      }
      env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
      }
      env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
      }
      env {
        name  = "GCP_CREDIT_USAGE_TOPIC"
        value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
      }
      env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
      env {
        name  = "UPLOADS_FOLDER"
        value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
      }
      env {
        name = "REDIS_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
            version = "latest"
          }
        }
      }

      env {
        name = "PSQL_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_USERNAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_DATABASE"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
            version = "latest"
          }
        }
      }

      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "PINECONE_INDEX_NAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
            version = "latest"
          }
        }
      }
      resources {
        limits = {
          cpu    = local.indexer_cpu
          memory = local.indexer_memory
        }
      }
    }
    timeout                          = "690s"
    max_instance_request_concurrency = local.indexer_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  depends_on = [
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_service.run_api,
    google_project_iam_member.secretaccessor
  ]
}

# Define Eventarc triggers for each region
resource "google_eventarc_trigger" "fileupload_trigger" {
  for_each = toset(local.us_regions)

  name     = "file-upload-trigger-${each.key}-${local.environment}"
  location = "us"

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }
  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.fileupload_bucket.name
  }

  service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.indexer_cloud_run[each.key].name
      region  = each.key
    }
  }
  depends_on = [
    google_storage_bucket.fileupload_bucket,
    google_pubsub_topic.fileupload_event_topic,
    google_cloud_run_v2_service.indexer_cloud_run
  ]
}

module "gcloud_pubsub_ack_deadline" {
  for_each = toset(local.us_regions)

  source  = "terraform-google-modules/gcloud/google"
  version = "~> 3.4"

  platform = "linux"

  create_cmd_entrypoint  = "gcloud"
  create_cmd_body        = "pubsub subscriptions update ${google_eventarc_trigger.fileupload_trigger[each.key].transport[0].pubsub[0].subscription} --ack-deadline 600 --min-retry-delay=590s --max-retry-delay=600s"
  destroy_cmd_entrypoint = "gcloud"
  destroy_cmd_body       = "pubsub subscriptions update ${google_eventarc_trigger.fileupload_trigger[each.key].transport[0].pubsub[0].subscription} --ack-deadline 600 --min-retry-delay=590s --max-retry-delay=600s"
}

## Searxng Cloud run externally hosted accessible via fe_backend
resource "google_cloud_run_service_iam_binding" "websearch_iam_binding" {
  for_each = toset(local.regions)

  location = each.key
  service  = google_cloud_run_v2_service.websearch_cloud_run[each.key].name
  role     = "roles/run.invoker"
  members = [
    "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com",
  ]
}

resource "google_cloud_run_v2_service" "websearch_cloud_run" {
  for_each = toset(local.regions)

  name     = "websearch${local.websearch_domain_suffix}-${each.key}"
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    scaling {
      max_instance_count = local.region_instance_counts[each.key].fe_max_inst
      min_instance_count = local.region_instance_counts[each.key].fe_min_inst
    }

    containers {
      ports {
        container_port = local.searxng_port # Example port, update as necessary
      }
      image = local.websearch_image # Update with your websearch image
      resources {
        limits = {
          cpu    = local.fe_cpu
          memory = local.fe_memory
        }
      }
    }
    timeout                          = "690s"
    max_instance_request_concurrency = local.fe_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  custom_audiences = [local.environment == "prod" ? "websearch.structhub.io" : "stage-websearch.structhub.io"]
  depends_on = [
    google_project_service.run_api
  ]
}

resource "google_compute_global_address" "websearch_external_ip" {
  name = "websearch${local.websearch_domain_suffix}-ip"
}

resource "google_compute_global_address" "xlsx_external_ip" {
  name = "xlsx${local.xlsx_domain_suffix}-ip"
}

resource "google_compute_backend_service" "websearch_backend" {
  name                            = "websearch${local.websearch_domain_suffix}-backend"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  connection_draining_timeout_sec = 310
  locality_lb_policy              = "RANDOM"
  enable_cdn                      = false

  dynamic "backend" {
    for_each = local.regions

    content {
      group = google_compute_region_network_endpoint_group.websearch_backend[backend.key].id
    }
  }

  depends_on = [
    google_project_service.compute_api,
  ]
}

resource "google_compute_backend_service" "xlsx_backend" {
  name                            = "xlsx${local.xlsx_domain_suffix}-backend"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  connection_draining_timeout_sec = 3600
  locality_lb_policy              = "RANDOM"
  enable_cdn                      = false

  dynamic "backend" {
    for_each = local.regions

    content {
      group = google_compute_region_network_endpoint_group.xlsx_backend[backend.key].id
    }
  }

  depends_on = [
    google_project_service.compute_api,
  ]
}

resource "google_compute_managed_ssl_certificate" "websearch_ssl_cert" {
  name = "websearch${local.websearch_domain_suffix}-cert"
  managed {
    domains = [local.environment == "prod" ? "websearch.structhub.io" : "stage-websearch.structhub.io"]
  }
}

resource "google_compute_target_https_proxy" "websearch_https_proxy" {
  name                        = "websearch${local.websearch_domain_suffix}-https"
  ssl_certificates            = [google_compute_managed_ssl_certificate.websearch_ssl_cert.id]
  url_map                     = google_compute_url_map.websearch_url_map.id
  http_keep_alive_timeout_sec = 610

  depends_on = [
    google_compute_managed_ssl_certificate.websearch_ssl_cert
  ]
}

resource "google_compute_url_map" "websearch_url_map" {
  name            = "websearch${local.websearch_domain_suffix}"
  default_service = google_compute_backend_service.websearch_backend.id
}

resource "google_compute_managed_ssl_certificate" "xlsx_ssl_cert" {
  name = "xlsx${local.xlsx_domain_suffix}-cert"
  managed {
    domains = [local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"]
  }
}

resource "google_compute_target_https_proxy" "xlsx_https_proxy" {
  name                        = "xlsx${local.xlsx_domain_suffix}-https"
  ssl_certificates            = [google_compute_managed_ssl_certificate.xlsx_ssl_cert.id]
  url_map                     = google_compute_url_map.xlsx_url_map.id
  http_keep_alive_timeout_sec = 610

  depends_on = [
    google_compute_managed_ssl_certificate.xlsx_ssl_cert
  ]
}

resource "google_compute_url_map" "xlsx_url_map" {
  name            = "xlsx${local.xlsx_domain_suffix}"
  default_service = google_compute_backend_service.xlsx_backend.id
}

resource "google_compute_global_forwarding_rule" "xlsx_forwarding_rule" {
  name                  = "xlsx${local.xlsx_domain_suffix}-https"
  target                = google_compute_target_https_proxy.xlsx_https_proxy.id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.xlsx_external_ip.id
  port_range            = "443"
  depends_on            = [google_compute_target_https_proxy.xlsx_https_proxy]
}

resource "google_compute_global_forwarding_rule" "websearch_forwarding_rule" {
  name                  = "websearch${local.websearch_domain_suffix}-https"
  target                = google_compute_target_https_proxy.websearch_https_proxy.id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.websearch_external_ip.id
  port_range            = "443"
  depends_on            = [google_compute_target_https_proxy.websearch_https_proxy]
}

resource "google_compute_region_network_endpoint_group" "websearch_backend" {
  count                 = length(local.regions)
  name                  = "websearch${local.websearch_domain_suffix}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.regions[count.index]
  cloud_run {
    service = google_cloud_run_v2_service.websearch_cloud_run[local.regions[count.index]].name
  }
}


## Confluence

# Create a Pub/Sub topic for Confluence
resource "google_pubsub_topic" "confluence_topic" {
  name = "confluence-topic-${local.environment}"
}

resource "google_pubsub_topic" "confluence_topic_dead_letter_topic" {
  name                       = "confluence-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}


resource "google_cloud_run_v2_job" "confluence_cloud_run_job" {
  # Deploy this job to multiple regions if desired
  for_each = toset(local.us_regions)

  # Name your job. You can tweak this however you like.
  name                = "confluence-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        # Remove the "ports" block entirely for a Cloud Run Job
        image = local.confluence_image

        # Pass environment variables/secrets into your container:
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        # Set resource limits
        resources {
          limits = {
            cpu    = local.confluence_cpu
            memory = local.confluence_memory
          }
        }
      }

      # Increase the timeout if you expect the job to run a while
      timeout = "10800s" # 3 hours
    }

    # For one-off ingestion, usually parallelism=1, task_count=1
    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}


# Generates an archive of the source code compressed as a .zip file.
data "archive_file" "confluence_topic_function_source" {
  type        = "zip"
  source_dir  = "../../confluence-topic-function"
  output_path = "${path.module}/confluence-topic-function.zip"
}

# Add source confluence topic function code zip to the Cloud Function's bucket (Cloud_function_bucket) 
resource "google_storage_bucket_object" "confluence_zip" {
  source       = "${path.module}/confluence-topic-function.zip"
  content_type = "application/zip"
  name         = "confluence-topic-function${local.fe_domain_suffix}-${data.archive_file.confluence_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.confluence_topic_function_bucket.name
  depends_on = [
    google_storage_bucket.confluence_topic_function_bucket,
    data.archive_file.confluence_topic_function_source
  ]
}

resource "google_storage_bucket" "confluence_topic_function_bucket" {
  name     = "confluence-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}
resource "google_cloud_run_service_iam_binding" "confluence_function_cloud_run_iam_binding" {
  project  = google_cloudfunctions2_function.confluence_trigger_function.project
  location = google_cloudfunctions2_function.confluence_trigger_function.location
  service  = google_cloudfunctions2_function.confluence_trigger_function.name
  role     = "roles/run.invoker"
  members = [
    "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com",
  ]
  depends_on = [google_cloudfunctions2_function.confluence_trigger_function]

  lifecycle {
    replace_triggered_by = [google_cloudfunctions2_function.confluence_trigger_function]
  }
}
resource "google_cloudfunctions2_function" "confluence_trigger_function" {
  name     = "confluence-topic-function-${local.environment}"
  location = "us-central1"
  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.confluence_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_confluence_job"
    source {
      storage_source {
        bucket = google_storage_bucket.confluence_topic_function_bucket.name
        object = "confluence-topic-function${local.fe_domain_suffix}-${data.archive_file.confluence_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true
    environment_variables = {
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.confluence_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }
  depends_on = [
    google_pubsub_topic.confluence_topic,
    google_pubsub_topic.confluence_topic_dead_letter_topic,
    google_cloud_run_v2_job.confluence_cloud_run_job
  ]
}


# Google Drive

# Create a Pub/Sub topic for Google drive Source
resource "google_pubsub_topic" "gdrive_topic" {
  name = "gdrive-topic-${local.environment}"
}

resource "google_pubsub_topic" "gdrive_topic_dead_letter_topic" {
  name                       = "gdrive-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}


resource "google_cloud_run_v2_job" "gdrive_cloud_run_job" {
  # Deploy this job to multiple regions if desired
  for_each = toset(local.us_regions)

  # Name your job. You can tweak this however you like.
  name                = "gdrive-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        # Remove the "ports" block entirely for a Cloud Run Job
        image = local.gdrive_image

        # Pass environment variables/secrets into your container:
        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }
        env {
          name = "GOOGLE_CLIENT_ID"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "GOOGLE_CLIENT_ID" : "GOOGLE_CLIENT_ID_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "GOOGLE_CLIENT_SECRET"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "GOOGLE_CLIENT_SECRET" : "GOOGLE_CLIENT_SECRET_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        # Set resource limits
        resources {
          limits = {
            cpu    = local.gdrive_cpu
            memory = local.gdrive_memory
          }
        }
      }

      # Increase the timeout if you expect the job to run a while
      timeout = "10800s" # 3 hours
    }

    # For one-off ingestion, usually parallelism=1, task_count=1
    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}


# Generates an archive of the source code compressed as a .zip file.
data "archive_file" "gdrive_topic_function_source" {
  type        = "zip"
  source_dir  = "../../gdrive-topic-function"
  output_path = "${path.module}/gdrive-topic-function.zip"
}

# Add source gdrive topic function code zip to the Cloud Function's bucket (Cloud_function_bucket) 
resource "google_storage_bucket_object" "gdrive_zip" {
  source       = "${path.module}/gdrive-topic-function.zip"
  content_type = "application/zip"
  name         = "gdrive-topic-function${local.fe_domain_suffix}-${data.archive_file.gdrive_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.gdrive_topic_function_bucket.name
  depends_on = [
    google_storage_bucket.gdrive_topic_function_bucket,
    data.archive_file.gdrive_topic_function_source
  ]
}

resource "google_storage_bucket" "gdrive_topic_function_bucket" {
  name     = "gdrive-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}
resource "google_cloud_run_service_iam_binding" "gdrive_function_cloud_run_iam_binding" {
  project  = google_cloudfunctions2_function.gdrive_trigger_function.project
  location = google_cloudfunctions2_function.gdrive_trigger_function.location
  service  = google_cloudfunctions2_function.gdrive_trigger_function.name
  role     = "roles/run.invoker"
  members = [
    "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com",
  ]
  depends_on = [google_cloudfunctions2_function.gdrive_trigger_function]

  lifecycle {
    replace_triggered_by = [google_cloudfunctions2_function.gdrive_trigger_function]
  }
}
resource "google_cloudfunctions2_function" "gdrive_trigger_function" {
  name     = "gdrive-topic-function-${local.environment}"
  location = "us-central1"
  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.gdrive_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_gdrive_job"
    source {
      storage_source {
        bucket = google_storage_bucket.gdrive_topic_function_bucket.name
        object = "gdrive-topic-function${local.fe_domain_suffix}-${data.archive_file.gdrive_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true
    environment_variables = {
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.gdrive_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }

    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }
  depends_on = [
    google_pubsub_topic.gdrive_topic,
    google_pubsub_topic.gdrive_topic_dead_letter_topic,
    google_cloud_run_v2_job.gdrive_cloud_run_job
  ]
}



#### ONE DRIVE

resource "google_pubsub_topic" "onedrive_topic" {
  name = "onedrive-topic-${local.environment}"
}

resource "google_pubsub_topic" "onedrive_topic_dead_letter_topic" {
  name                       = "onedrive-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s" # ~31 days, adjust as needed
}

resource "google_cloud_run_v2_job" "onedrive_cloud_run_job" {
  # Deploy this job to multiple regions if desired
  for_each = toset(local.us_regions)

  name                = "onedrive-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        # NOTE: No ports for a Cloud Run Job container
        image = local.onedrive_image

        # Environment variables / secrets
        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        # Insert your OneDrive-specific secrets or config here:
        env {
          name = "ONEDRIVE_CLIENT_ID"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "ONEDRIVE_CLIENT_ID" : "ONEDRIVE_CLIENT_ID_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "ONEDRIVE_CLIENT_SECRET"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "ONEDRIVE_CLIENT_SECRET" : "ONEDRIVE_CLIENT_SECRET_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = local.gdrive_cpu    # or local.onedrive_cpu if you have a separate variable
            memory = local.gdrive_memory # or local.onedrive_memory
          }
        }
      }

      # Adjust the Job timeout as needed for your OneDrive ingestion tasks
      timeout = "10800s" # 3 hours
    }

    # For a one-off ingestion, typically parallelism=1, task_count=1
    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}


data "archive_file" "onedrive_topic_function_source" {
  type        = "zip"
  source_dir  = "../../onedrive-topic-function"
  output_path = "${path.module}/onedrive-topic-function.zip"
}

resource "google_storage_bucket" "onedrive_topic_function_bucket" {
  name     = "onedrive-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "onedrive_zip" {
  source       = "${path.module}/onedrive-topic-function.zip"
  content_type = "application/zip"
  name         = "onedrive-topic-function${local.fe_domain_suffix}-${data.archive_file.onedrive_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.onedrive_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.onedrive_topic_function_bucket,
    data.archive_file.onedrive_topic_function_source
  ]
}

resource "google_cloudfunctions2_function" "onedrive_trigger_function" {
  name     = "onedrive-topic-function-${local.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.onedrive_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_onedrive_job" # Adjust to match your Python function name
    source {
      storage_source {
        bucket = google_storage_bucket.onedrive_topic_function_bucket.name
        object = "onedrive-topic-function${local.fe_domain_suffix}-${data.archive_file.onedrive_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true

    environment_variables = {
      # The function typically needs to know the Cloud Run Job name/region it must invoke
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.onedrive_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }

    # If your function needs DB credentials or other secrets
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }

  depends_on = [
    google_pubsub_topic.onedrive_topic,
    google_pubsub_topic.onedrive_topic_dead_letter_topic,
    google_cloud_run_v2_job.onedrive_cloud_run_job
  ]
}

# create a subscription to the dead-letter topic
resource "google_pubsub_subscription" "onedrive_dead_letter_subscription" {
  name  = "onedrive_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.onedrive_topic_dead_letter_topic.name
}

#### SHAREPOINT

resource "google_pubsub_topic" "sharepoint_topic" {
  name = "sharepoint-topic-${local.environment}"
}


resource "google_pubsub_topic" "sharepoint_topic_dead_letter_topic" {
  name                       = "sharepoint-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s" # ~31 days, adjust as needed
}


resource "google_cloud_run_v2_job" "sharepoint_cloud_run_job" {
  # Deploy this job to multiple regions if desired
  for_each = toset(local.us_regions)

  name                = "sharepoint-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        # NOTE: No ports for a Cloud Run Job container
        image = local.sharepoint_image

        # Environment variables / secrets
        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        # Insert your SharePoint-specific secrets or config here:
        env {
          name = "SHAREPOINT_CLIENT_ID"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SHAREPOINT_CLIENT_ID" : "SHAREPOINT_CLIENT_ID_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SHAREPOINT_CLIENT_SECRET"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SHAREPOINT_CLIENT_SECRET" : "SHAREPOINT_CLIENT_SECRET_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }

        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = local.sharepoint_cpu    # Define local.sharepoint_cpu accordingly
            memory = local.sharepoint_memory # Define local.sharepoint_memory accordingly
          }
        }
      }

      # Adjust the Job timeout as needed for your SharePoint ingestion tasks
      timeout = "10800s" # 3 hours
    }

    # For a one-off ingestion, typically parallelism=1, task_count=1
    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}


data "archive_file" "sharepoint_topic_function_source" {
  type        = "zip"
  source_dir  = "../../sharepoint-topic-function" # Adjust the path as necessary
  output_path = "${path.module}/sharepoint-topic-function.zip"
}

resource "google_storage_bucket" "sharepoint_topic_function_bucket" {
  name     = "sharepoint-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "sharepoint_zip" {
  source       = "${path.module}/sharepoint-topic-function.zip"
  content_type = "application/zip"
  name         = "sharepoint-topic-function${local.fe_domain_suffix}-${data.archive_file.sharepoint_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.sharepoint_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.sharepoint_topic_function_bucket,
    data.archive_file.sharepoint_topic_function_source
  ]
}


resource "google_cloudfunctions2_function" "sharepoint_trigger_function" {
  name     = "sharepoint-topic-function-${local.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.sharepoint_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_sharepoint_job" # Adjust to match your Python function name
    source {
      storage_source {
        bucket = google_storage_bucket.sharepoint_topic_function_bucket.name
        object = "sharepoint-topic-function${local.fe_domain_suffix}-${data.archive_file.sharepoint_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true

    environment_variables = {
      # The function typically needs to know the Cloud Run Job name/region it must invoke
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.sharepoint_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }

    # If your function needs DB credentials or other secrets
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }

  depends_on = [
    google_pubsub_topic.sharepoint_topic,
    google_pubsub_topic.sharepoint_topic_dead_letter_topic,
    google_cloud_run_v2_job.sharepoint_cloud_run_job
  ]
}

# Create a subscription to the dead-letter topic
resource "google_pubsub_subscription" "sharepoint_dead_letter_subscription" {
  name  = "sharepoint_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.sharepoint_topic_dead_letter_topic.name
}

############################
# S3 Pub/Sub Topics
############################
resource "google_pubsub_topic" "s3_topic" {
  name = "s3-topic-${local.environment}"
}

resource "google_pubsub_topic" "s3_topic_dead_letter_topic" {
  name                       = "s3-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s" # ~31 days
}

resource "google_cloud_run_v2_job" "s3_cloud_run_job" {
  # Deploy to multiple regions if you want
  for_each = toset(local.us_regions)

  name                = "s3-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        image = local.s3_image # Docker image for your s3_ingest.py job

        # Example env vars

        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        # Example secrets
        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = local.s3_cpu    # define local.s3_cpu
            memory = local.s3_memory # define local.s3_memory
          }
        }
      }

      timeout = "10800s" # 3 hours, for example
    }

    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}

data "archive_file" "s3_topic_function_source" {
  type        = "zip"
  source_dir  = "../../s3-topic-function" # path to your CF code
  output_path = "${path.module}/s3-topic-function.zip"
}

resource "google_storage_bucket" "s3_topic_function_bucket" {
  name     = "s3-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "s3_zip" {
  source       = "${path.module}/s3-topic-function.zip"
  content_type = "application/zip"
  name         = "s3-topic-function${local.fe_domain_suffix}-${data.archive_file.s3_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.s3_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.s3_topic_function_bucket,
    data.archive_file.s3_topic_function_source
  ]
}

resource "google_cloudfunctions2_function" "s3_trigger_function" {
  name     = "s3-topic-function-${local.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.s3_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_s3_job" # your CF python function
    source {
      storage_source {
        bucket = google_storage_bucket.s3_topic_function_bucket.name
        object = "s3-topic-function${local.fe_domain_suffix}-${data.archive_file.s3_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true

    environment_variables = {
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.s3_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }

  depends_on = [
    google_pubsub_topic.s3_topic,
    google_pubsub_topic.s3_topic_dead_letter_topic,
    google_cloud_run_v2_job.s3_cloud_run_job
  ]
}

resource "google_pubsub_subscription" "s3_dead_letter_subscription" {
  name  = "s3_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.s3_topic_dead_letter_topic.name
}


# GCP 
resource "google_pubsub_topic" "gcpbucket_topic" {
  name = "gcpbucket-topic-${local.environment}"
}

resource "google_pubsub_topic" "gcpbucket_topic_dead_letter_topic" {
  name                       = "gcpbucket-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}

resource "google_cloud_run_v2_job" "gcpbucket_cloud_run_job" {
  for_each = toset(local.us_regions)

  name                = "gcpbucket-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        image = local.gcpbucket_image

        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }

        resources {
          limits = {
            cpu    = local.gcpbucket_cpu
            memory = local.gcpbucket_memory
          }
        }
      }

      timeout = "10800s"
    }

    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}

data "archive_file" "gcpbucket_topic_function_source" {
  type        = "zip"
  source_dir  = "../../gcpBucket-topic-function"
  output_path = "${path.module}/gcpbucket-topic-function.zip"
}

resource "google_storage_bucket" "gcpbucket_topic_function_bucket" {
  name     = "gcpbucket-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "gcpbucket_zip" {
  source       = "${path.module}/gcpbucket-topic-function.zip"
  content_type = "application/zip"
  name         = "gcpbucket-topic-function${local.fe_domain_suffix}-${data.archive_file.gcpbucket_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.gcpbucket_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.gcpbucket_topic_function_bucket,
    data.archive_file.gcpbucket_topic_function_source
  ]
}

resource "google_cloudfunctions2_function" "gcpbucket_trigger_function" {
  name     = "gcpbucket-topic-function-${local.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.gcpbucket_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_gcpBucket_job"
    source {
      storage_source {
        bucket = google_storage_bucket.gcpbucket_topic_function_bucket.name
        object = "gcpbucket-topic-function${local.fe_domain_suffix}-${data.archive_file.gcpbucket_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true

    environment_variables = {
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.gcpbucket_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }

  depends_on = [
    google_pubsub_topic.gcpbucket_topic,
    google_pubsub_topic.gcpbucket_topic_dead_letter_topic,
    google_cloud_run_v2_job.gcpbucket_cloud_run_job
  ]
}

resource "google_pubsub_subscription" "gcpbucket_dead_letter_subscription" {
  name  = "gcpbucket_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.gcpbucket_topic_dead_letter_topic.name
}

# BM25 vocab updater function:
resource "google_pubsub_topic" "bm25_vocab_updater_topic" {
  name = "bm25-vocab-updater-topic-${local.environment}"
}

resource "google_pubsub_topic" "bm25_vocab_updater_topic_dead_letter_topic" {
  name                       = "bm25-vocab-updater-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}

data "archive_file" "bm25_vocab_updater_topic_function_source" {
  type        = "zip"
  source_dir  = "../../bm25-vocab-updater-function"
  output_path = "${path.module}/bm25-vocab-updater-topic-function.zip"
}

resource "google_storage_bucket" "bm25_vocab_updater_topic_function_bucket" {
  name     = "bm25-vocab-updater-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "bm25_vocab_updater_zip" {
  source       = "${path.module}/bm25-vocab-updater-topic-function.zip"
  content_type = "application/zip"
  name         = "bm25-vocab-updater-topic-function${local.fe_domain_suffix}-${data.archive_file.bm25_vocab_updater_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.bm25_vocab_updater_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.bm25_vocab_updater_topic_function_bucket,
    data.archive_file.bm25_vocab_updater_topic_function_source
  ]
}


resource "google_cloudfunctions2_function" "bm25_vocab_updater" {
  name     = "bm25-vocab-updater-${var.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.bm25_vocab_updater_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_bm25_vocab_updater"
    source {
      storage_source {
        bucket = google_storage_bucket.bm25_vocab_updater_topic_function_bucket.name
        object = google_storage_bucket_object.bm25_vocab_updater_zip.name
      }
    }
  }

  service_config {
    max_instance_count  = 10000
    min_instance_count = 1
    available_memory               = "256M"
    timeout_seconds                = 300
    all_traffic_on_latest_revision = true
    environment_variables = {
      GCP_PROJECT_ID = local.project_id,
      BM25_VOCAB_UPDATES_TOPIC = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
      FIRESTORE_DB = google_firestore_database.firestore.name
    }
  }

  depends_on = [
    google_pubsub_topic.bm25_vocab_updater_topic,
    google_storage_bucket_object.bm25_vocab_updater_zip,
    google_firestore_database.firestore
  ]
}

resource "google_pubsub_subscription" "bm25_vocab_updater_dead_letter_subscription" {
  name  = "bm25_vocab_updater_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.bm25_vocab_updater_topic_dead_letter_topic.name
}

############################
# Azure Blob Pub/Sub Topics
############################
resource "google_pubsub_topic" "azureblob_topic" {
  name = "azureblob-topic-${local.environment}"
}

resource "google_pubsub_topic" "azureblob_topic_dead_letter_topic" {
  name                       = "azureblob-topic-dead-letter-topic${local.fe_domain_suffix}"
  message_retention_duration = "2678400s"
}

resource "google_cloud_run_v2_job" "azureblob_cloud_run_job" {
  for_each = toset(local.us_regions)

  name                = "azureblob-job${local.indexer_domain_suffix}-${each.key}"
  location            = each.key

  template {
    template {
      service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

      containers {
        image = local.azureblob_image

        env {
          name  = "SERVER_URL"
          value = local.environment == "prod" ? "be.api.structhub.io" : "stage-be.api.structhub.io"
        }
        env {
        name  = "XLSX_SERVER_URL"
        value = local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"
        }
        env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
        }
        env {
          name  = "UPLOADS_FOLDER"
          value = "/app/uploads"
        }
        env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }
        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }
        resources {
          limits = {
            cpu    = local.azureblob_cpu
            memory = local.azureblob_memory
          }
        }
      }

      timeout = "10800s"
    }

    parallelism = 1
    task_count  = 1
  }

  depends_on = [
    google_project_service.run_api,
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_iam_member.secretaccessor
  ]
}

data "archive_file" "azureblob_topic_function_source" {
  type        = "zip"
  source_dir  = "../../azureBlob-topic-function"
  output_path = "${path.module}/azureblob-topic-function.zip"
}

resource "google_storage_bucket" "azureblob_topic_function_bucket" {
  name     = "azureblob-function-bucket${local.fe_domain_suffix}"
  location = "us-central1"
}

resource "google_storage_bucket_object" "azureblob_zip" {
  source       = "${path.module}/azureblob-topic-function.zip"
  content_type = "application/zip"
  name         = "azureblob-topic-function${local.fe_domain_suffix}-${data.archive_file.azureblob_topic_function_source.output_md5}.zip"
  bucket       = google_storage_bucket.azureblob_topic_function_bucket.name

  depends_on = [
    google_storage_bucket.azureblob_topic_function_bucket,
    data.archive_file.azureblob_topic_function_source
  ]
}

resource "google_cloudfunctions2_function" "azureblob_trigger_function" {
  name     = "azureblob-topic-function-${local.environment}"
  location = "us-central1"

  event_trigger {
    event_type     = "google.cloud.pubsub.topic.v1.messagePublished"
    pubsub_topic   = google_pubsub_topic.azureblob_topic.id
    trigger_region = "us-central1"
    retry_policy   = "RETRY_POLICY_RETRY"
  }

  build_config {
    runtime     = "python310"
    entry_point = "pubsub_to_cloud_run_azureBlob_job"
    source {
      storage_source {
        bucket = google_storage_bucket.azureblob_topic_function_bucket.name
        object = "azureblob-topic-function${local.fe_domain_suffix}-${data.archive_file.azureblob_topic_function_source.output_md5}.zip"
      }
    }
  }

  service_config {
    available_memory               = "256M"
    max_instance_count             = 10000
    timeout_seconds                = 60
    all_traffic_on_latest_revision = true

    environment_variables = {
      CLOUD_RUN_JOB_NAME = google_cloud_run_v2_job.azureblob_cloud_run_job["us-central1"].name
      CLOUD_RUN_REGION   = "us-central1"
      GCP_PROJECT_ID     = local.project_id
    }
    secret_environment_variables {
      project_id = local.project_id
      secret     = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
      key        = "PSQL_HOST"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PASSWORD"
      secret     = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_USERNAME"
      secret     = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_DATABASE"
      secret     = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
      version    = "latest"
    }
    secret_environment_variables {
      project_id = local.project_id
      key        = "PSQL_PORT"
      secret     = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
      version    = "latest"
    }
  }

  depends_on = [
    google_pubsub_topic.azureblob_topic,
    google_pubsub_topic.azureblob_topic_dead_letter_topic,
    google_cloud_run_v2_job.azureblob_cloud_run_job
  ]
}

resource "google_pubsub_subscription" "azure_dead_letter_subscription" {
  name  = "azureblob_dead_letter_subscription${local.fe_domain_suffix}"
  topic = google_pubsub_topic.azureblob_topic_dead_letter_topic.name
}

# New xlsx-indexer service
resource "google_cloud_run_v2_service" "xlsx_cloud_run" {
  for_each = toset(local.regions)

  name     = "${local.xlsx_service_name_prefix}${local.xlsx_domain_suffix}-${each.key}"
  
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    scaling {
      max_instance_count = local.region_instance_counts[each.key].be_max_inst
      min_instance_count = local.region_instance_counts[each.key].be_min_inst
    }
    containers {
      env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }
      ports {
        container_port = local.xlsx_port
      }
      image = local.xlsx_image
      resources {
        limits = {
          cpu    = local.xlsx_cpu
          memory = local.xlsx_memory
        }
      }
    }
    timeout                          = "3600s"  # 1 hour for xlsx-indexer
    max_instance_request_concurrency = 1  # Recommended for long-running tasks
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  custom_audiences = [local.environment == "prod" ? "xlsx.structhub.io" : "stage-xlsx.structhub.io"]
  depends_on = [
    google_project_service.run_api
  ]
}


# Metadata tag service:

resource "google_compute_region_network_endpoint_group" "metadata_backend" {
  count                 = length(local.regions)
  name                  = "${local.metadata_service_name_prefix}${local.metadata_domain_suffix}-neg"
  network_endpoint_type = "SERVERLESS"
  region                = local.regions[count.index]
  cloud_run {
    service = google_cloud_run_v2_service.metadata_cloud_run[local.regions[count.index]].name
  }
}


resource "google_compute_global_address" "metadata_external_ip" {
  name = "metadata${local.metadata_domain_suffix}-ip"
}

resource "google_compute_backend_service" "metadata_backend" {
  name                            = "metadata${local.metadata_domain_suffix}-backend"
  load_balancing_scheme           = "EXTERNAL_MANAGED"
  connection_draining_timeout_sec = 3600
  locality_lb_policy              = "RANDOM"
  enable_cdn                      = false

  dynamic "backend" {
    for_each = local.regions

    content {
      group = google_compute_region_network_endpoint_group.metadata_backend[backend.key].id
    }
  }

  depends_on = [
    google_project_service.compute_api,
  ]
}


resource "google_compute_managed_ssl_certificate" "metadata_ssl_cert" {
  name = "metadata${local.metadata_domain_suffix}-cert"
  managed {
    domains = [local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"]
  }
}

resource "google_compute_target_https_proxy" "metadata_https_proxy" {
  name                        = "metadata${local.metadata_domain_suffix}-https"
  ssl_certificates            = [google_compute_managed_ssl_certificate.metadata_ssl_cert.id]
  url_map                     = google_compute_url_map.metadata_url_map.id
  http_keep_alive_timeout_sec = 610

  depends_on = [
    google_compute_managed_ssl_certificate.metadata_ssl_cert
  ]
}

resource "google_compute_url_map" "metadata_url_map" {
  name            = "metadata${local.metadata_domain_suffix}"
  default_service = google_compute_backend_service.metadata_backend.id
}

resource "google_compute_global_forwarding_rule" "metadata_forwarding_rule" {
  name                  = "metadata${local.metadata_domain_suffix}-https"
  target                = google_compute_target_https_proxy.metadata_https_proxy.id
  load_balancing_scheme = "EXTERNAL_MANAGED"
  ip_address            = google_compute_global_address.metadata_external_ip.id
  port_range            = "443"
  depends_on            = [google_compute_target_https_proxy.metadata_https_proxy]
}

resource "google_cloud_run_v2_service" "metadata_cloud_run" {
  for_each = toset(local.regions)
  
  name     = "${local.metadata_service_name_prefix}${local.metadata_domain_suffix}-${each.key}"
  
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    scaling {
      max_instance_count = local.region_instance_counts[each.key].be_max_inst
      min_instance_count = local.region_instance_counts[each.key].be_min_inst
    }
    containers {
      env {
          name  = "GCP_PROJECT_ID"
          value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
        }
        env {
        name  = "METADATA_FILEUPLOAD_BUCKET"
        value = google_storage_bucket.metadata_fileupload_bucket.name
      }
        env {
          name  = "GCP_CREDIT_USAGE_TOPIC"
          value = "structhub-credit-usage-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "BM25_VOCAB_UPDATES_TOPIC"
          value = "bm25-vocab-updater-topic${local.fe_domain_suffix}"
        }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
        env {
          name  = "ENVIRONMENT"
          value = local.environment
        }

        env {
          name = "PSQL_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_USERNAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_DATABASE"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PSQL_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "SECRET_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_HOST"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PASSWORD"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "REDIS_PORT"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_API_KEY"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
              version = "latest"
            }
          }
        }
        env {
          name = "PINECONE_INDEX_NAME"
          value_source {
            secret_key_ref {
              secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
              version = "latest"
            }
          }
        }
      ports {
        container_port = local.metadata_port
      }
      image = local.metadata_image
      resources {
        limits = {
          cpu    = local.metadata_cpu
          memory = local.metadata_memory
        }
      }
    }
    timeout                          = "3600s"  # 1 hour for metadata-indexer
    max_instance_request_concurrency = 1  # Recommended for long-running tasks
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  custom_audiences = [local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"]
  depends_on = [
    google_project_service.run_api
  ]
}


resource "google_storage_bucket" "metadata_fileupload_bucket" {
  name                        = "structhub-metadata-file-upload-bucket-${local.environment}"
  location                    = "us"
  uniform_bucket_level_access = true
  cors {
    origin          = ["https://stage.structhub.io", "http://localhost:3000", "https://structhub.io"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

resource "google_storage_bucket_iam_binding" "metadata_bucket_access" {
  bucket = google_storage_bucket.metadata_fileupload_bucket.name

  role    = "roles/storage.objectAdmin"
    members = [
    "serviceAccount:xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com",
  ]
}

# resource "google_service_account" "metadata_sa" {
#   account_id   = "${local.environment}-metadata-sa"
#   display_name = "Metadata Cloud Run SA (${local.environment})"
# }


## AUTO Metadata keys:

resource "google_storage_bucket" "metadata_keys_bucket" {
  name                        = "structhub-metadata-keys-upload-bucket-${local.environment}"
  location                    = "us"
  uniform_bucket_level_access = true
  cors {
    origin          = ["https://stage.structhub.io", "http://localhost:3000", "https://structhub.io"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

resource "google_cloud_run_v2_service" "metadata_keys_cloud_run" {
  for_each = toset(local.us_regions)

  name     = "${local.metadata_keys_service_name_prefix}${local.metadata_keys_domain_suffix}-${each.key}"
  location = each.key
  template {
    service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"
    scaling {
      max_instance_count = local.region_instance_counts[each.key].indexer_max_inst
      min_instance_count = local.region_instance_counts[each.key].indexer_min_inst
    }

    containers {
      ports {
        container_port = local.metadata_keys_port
      }
      image = local.metadata_keys_image
      env {
        name  = "METADATA_SERVER_URL"
        value = local.environment == "prod" ? "metadata.structhub.io" : "stage-metadata.structhub.io"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = local.environment == "prod" ? "structhub-412620" : "structhub-412620"
      }
      env {
        name  = "GCP_CREDIT_USAGE_TOPIC"
        value = "structhub-credit-usage-topic${local.metadata_keys_domain_suffix}"
      }
        env {
          name  = "FIRESTORE_DB"
          value = google_firestore_database.firestore.name
        }
      env {
        name  = "UPLOADS_FOLDER"
        value = local.environment == "prod" ? "/app/uploads" : "/app/uploads"
      }
      env {
        name = "REDIS_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_HOST" : "REDIS_HOST_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PASSWORD" : "REDIS_PASSWORD_STAGE"
            version = "latest"
          }
        }
      }

      env {
        name = "PSQL_HOST"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_HOST" : "PSQL_HOST_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PASSWORD" : "PSQL_PASSWORD_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_USERNAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_USERNAME" : "PSQL_USERNAME_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PSQL_DATABASE"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_DATABASE" : "PSQL_DATABASE_STAGE"
            version = "latest"
          }
        }

      }

      env {
        name = "PSQL_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PSQL_PORT" : "PSQL_PORT_STAGE"
            version = "latest"
          }
        }

      }
      env {
        name = "PINECONE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_API_KEY" : "PINECONE_API_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "PINECONE_INDEX_NAME"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "PINECONE_INDEX_NAME" : "PINECONE_INDEX_NAME_STAGE"
            version = "latest"
          }
        }
      }

      env {
        name = "SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "SECRET_KEY" : "SECRET_KEY_STAGE"
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_PORT"
        value_source {
          secret_key_ref {
            secret  = local.environment == "prod" ? "REDIS_PORT" : "REDIS_PORT_STAGE"
            version = "latest"
          }
        }
      }
      resources {
        limits = {
          cpu    = local.metadata_keys_cpu
          memory = local.metadata_keys_memory
        }
      }
    }
    timeout                          = "690s"
    max_instance_request_concurrency = local.indexer_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  depends_on = [
    google_project_iam_member.indexer_eventreceiver,
    google_project_iam_member.indexer_runinvoker,
    google_project_iam_member.indexer_pubsubpublisher,
    google_project_service.run_api,
    google_project_iam_member.secretaccessor
  ]
}
resource "google_pubsub_topic" "metadata_keys_fileupload_event_topic" {
  name = "metadata-keys-file-upload-event-topic-${local.environment}"
}
# Define Eventarc triggers for each region
resource "google_eventarc_trigger" "metdata_keys_fileupload_trigger" {
  for_each = toset(local.us_regions)

  name     = "metdata-keys-file-upload-trigger-${each.key}-${local.environment}"
  location = "us"

  matching_criteria {
    attribute = "type"
    value     = "google.cloud.storage.object.v1.finalized"
  }
  matching_criteria {
    attribute = "bucket"
    value     = google_storage_bucket.metadata_keys_bucket.name
  }

  service_account = "xtract-fe-service-account@structhub-412620.iam.gserviceaccount.com"

  destination {
    cloud_run_service {
      service = google_cloud_run_v2_service.metadata_keys_cloud_run[each.key].name
      region  = each.key
    }
  }
  depends_on = [
    google_storage_bucket.metadata_keys_bucket,
    google_pubsub_topic.metadata_keys_fileupload_event_topic,
    google_cloud_run_v2_service.metadata_keys_cloud_run
  ]
}