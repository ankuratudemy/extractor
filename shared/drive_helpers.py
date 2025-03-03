#!/usr/bin/env python3
# drive_helpers.py

import requests
import io
from datetime import datetime
from typing import List, Dict
from zoneinfo import ZoneInfo

from dateutil import parser
from shared.logging_config import log

CENTRAL_TZ = ZoneInfo("America/Chicago")


def _list_all_files_recursive(token: str, folder_id: str) -> List[dict]:
    """
    Recursively list all files (including subfolders) in the given `folder_id` on Google Drive.
    Uses v3: GET https://www.googleapis.com/drive/v3/files
        params: q, fields, pageToken, pageSize
    """
    results = []
    stack = [folder_id]
    base_url = "https://www.googleapis.com/drive/v3/files"
    headers = {"Authorization": f"Bearer {token}"}
    fields = "files(id,name,mimeType,createdTime,modifiedTime,trashed),nextPageToken"

    while stack:
        current_folder = stack.pop()
        query = f"'{current_folder}' in parents and trashed=false"
        page_token = None

        while True:
            params = {
                "q": query,
                "fields": fields,
                "pageSize": 1000
            }
            if page_token:
                params["pageToken"] = page_token

            log.debug(f"[_list_all_files_recursive] GET => {base_url}, folder={current_folder}, pageToken={page_token}")
            resp = requests.get(base_url, headers=headers, params=params)
            if not resp.ok:
                resp_snip = resp.text[:300]
                log.error(f"[_list_all_files_recursive] Listing files failed: {resp.status_code}, {resp_snip}")
                if resp.status_code == 401:
                    # Let caller handle token refresh
                    raise ValueError("401 Unauthorized => needs token refresh")
                resp.raise_for_status()

            data = resp.json()
            files = data.get("files", [])
            for f in files:
                # If folder => push to stack
                if f.get("mimeType") == "application/vnd.google-apps.folder":
                    stack.append(f["id"])
                results.append(f)

            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return results


def list_all_files_recursively_with_retry(access_token: str, folder_id: str, ds) -> List[dict]:
    """
    Calls `_list_all_files_recursive`, letting the caller handle token refresh logic outside.
    """
    log.info("[list_all_files_recursively_with_retry] Listing Google Drive files with token.")
    try:
        drive_files = _list_all_files_recursive(access_token, folder_id)
        return drive_files
    except Exception as e:
        log.error(f"list_all_files_recursively_with_retry => failed to list: {str(e)}")
        raise


def download_drive_file_content(access_token: str, file_id: str, mime_type: str = None):
    """
    If the file is a Google Docs/Sheets/Slides => we must export it.
    Otherwise, alt=media for standard binary files.
    """
    from shared.google_mime_map import GDRIVE_MIME_EXT_MAP  # e.g. your dictionary
    if mime_type and mime_type.startswith("application/vnd.google-apps."):
        # It's a Google Doc/Sheet/Slide => we choose the right export
        export_ext = GDRIVE_MIME_EXT_MAP.get(mime_type, "pdf")
        if export_ext == "docx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif export_ext == "xlsx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif export_ext == "pptx":
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/vnd.openxmlformats-officedocument.presentationml.presentation"
        else:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}/export?mimeType=application/pdf"
            export_ext = "pdf"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()
        return (r.content, export_ext)
    else:
        # alt=media approach
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(url, headers=headers, stream=True)
        r.raise_for_status()

        # If we have a known extension from your map:
        ext = "NA"
        if mime_type:
            ext = GDRIVE_MIME_EXT_MAP.get(mime_type, "NA")
        return (r.content, ext)
