terraform {
  backend "gcs" {
    bucket  = "structhub_terraform_state_bucket_stage"
    prefix  = "terraform/state/"
 }
}