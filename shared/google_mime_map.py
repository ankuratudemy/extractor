#!/usr/bin/env python3
# google_mime_map.py

"""
A dictionary to map Google Drive MIME types (especially the Google Apps 'application/vnd.google-apps.*')
to a preferred export format extension.

If the MIME type is unknown or we want a default fallback, we can set "pdf".
"""

GDRIVE_MIME_EXT_MAP = {
    # If we see these custom vnd.google-apps.* MIME types, we choose an extension that we want to export as:
    "application/vnd.google-apps.document": "docx",
    "application/vnd.google-apps.spreadsheet": "xlsx",
    "application/vnd.google-apps.presentation": "pptx",

    # If you want to treat Drawings as PDF or PNG:
    "application/vnd.google-apps.drawing": "pdf",

    # Some other vnd.google-apps.* placeholders:
    "application/vnd.google-apps.form": "pdf",
    "application/vnd.google-apps.script": "pdf",
    "application/vnd.google-apps.site": "pdf",
    # ... add more as desired ...

    # If you have standard MIME types already recognized, you can fill them in:
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",

    # For anything else, you might fallback or define "NA" for "not available"
    # but in your code, you might default to "pdf" if you want a single fallback.
    # For example:
    # "default": "pdf"
}
