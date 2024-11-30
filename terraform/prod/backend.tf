terraform {
  backend "gcs" {
    bucket  = "structhub_terraform_state_bucket"
    prefix  = "terraform/state/"
 }
}