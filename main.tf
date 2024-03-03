provider "google" {
  # credentials = file("/dev/null")
  project     = "structhub-412620"
  region      = "us-central1"
}

variable "environment" {
  description = "Environment: 'stage' or 'prod'"
  type = string
  default     = "stage"
}

locals {
  environment             = var.environment  # Set the desired environment here
  regions                 = ["northamerica-northeast1"]
  fe_max_inst             = 40
  fe_min_inst             = 0
  fe_cpu                  = 1
  fe_memory               = "2Gi"
  fe_port                 = 5000
  be_max_inst             = 60
  be_min_inst             = 0
  be_cpu                  = 1
  be_memory               = "2Gi"
  be_port                 = 9998
  external_ip_address_name_fe = "xtract-fe-ip-name"
  external_ip_address_name_be = "xtract-be-ip-name"
  be_image                = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-be:1.0.0"
  fe_image                = "us-central1-docker.pkg.dev/structhub-412620/xtract/xtract-fe:gcr-53.0.0"
  be_concurrent_requests_per_inst = 1
  fe_concurrent_requests_per_inst = 1
  project_id              = "structhub-412620"
  fe_service_name_prefix  = "xtract-fe"
  be_service_name_prefix  = "xtract-be"
  fe_hc_path              = "/health"
  be_hc_path              = "/tika"
  fe_domain_suffix        = local.environment == "prod" ? "" : "-stage"
  be_domain_suffix        = local.environment == "prod" ? "" : "-stage"
}

resource "google_project_service" "compute_api" {
  service                    = "compute.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

resource "google_project_service" "run_api" {
  service                    = "run.googleapis.com"
  disable_dependent_services = false
  disable_on_destroy         = false
}

# resource "google_compute_address" "external_ip_fe" {
#   name = local.external_ip_address_name_fe
# }

# resource "google_compute_address" "external_ip_be" {
#   name = local.external_ip_address_name_be
# }

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
  connection_draining_timeout_sec = 70
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
    domains = [local.environment == "prod" ? "api.structhub.io" : "stage-be.api.structhub.io"]
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

resource "google_cloud_run_v2_service" "fe_cloud_run" {
  for_each = toset(local.regions)

  name             = "${local.fe_service_name_prefix}${local.fe_domain_suffix}-${each.key}"
  location         = each.key
  ingress          = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  custom_audiences = [local.environment == "prod" ? "api.structhub.io" : "stage-be.api.structhub.io"]
  template {
    scaling {
      max_instance_count = local.fe_max_inst
      min_instance_count = local.fe_min_inst
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
        name = "SERVER_URL"
        value = local.environment == "prod" ? "api.structhub.io" : "stage-be.api.structhub.io"
      }
      env {
        name = "REDIS_HOST"
        value_source {
          secret_key_ref {
            secret  = "REDIS_HOST"
            version = "1"
          }
        }
      }
      env {
        name = "REDIS_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = "REDIS_PASSWORD"
            version = "1"
          }
        }
      }
      env {
        name = "SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = "SECRET_KEY"
            version = "1"
          }
        }
      }
      env {
        name = "REDIS_PORT"
        value_source {
          secret_key_ref {
            secret  = "REDIS_PORT"
            version = "1"
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

  name     = "${local.be_service_name_prefix}${local.fe_domain_suffix}-${each.key}"
  location = each.key
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"
  template {
    scaling {
      max_instance_count = local.be_max_inst
      min_instance_count = local.be_min_inst
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
    timeout                          = "110s"
    max_instance_request_concurrency = local.be_concurrent_requests_per_inst
  }
  traffic {
    percent = 100
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
  }
  custom_audiences = [local.environment == "prod" ? "api.structhub.io" : "stage-be.api.structhub.io"]
  depends_on = [
    google_project_service.run_api
  ]
}
