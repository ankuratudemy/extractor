#!/usr/bin/env python3
# sharepoint_helpers.py

import os
import io
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from zoneinfo import ZoneInfo
from dateutil import parser

from shared.logging_config import log

# If you want to unify these next lines with common_code.py, you can do so.
CENTRAL_TZ = ZoneInfo("America/Chicago")


def _list_site_based(token: str, site_id: str, folder_id: str) -> List[dict]:
    """
    Site-based approach:
      GET /sites/{siteId}/drive/items/{folder_id}/children => returns items in that folder.
      We'll gather everything in a stack for subfolders (recursively).
    """
    base_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{folder_id}/children?$top=200"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    stack = [base_url]
    while stack:
        url = stack.pop()
        log.debug(f"[_list_site_based] GET => {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # Might raise HTTPError (400, 404, etc.)
        data = resp.json()
        items = data.get("value", [])
        for item in items:
            # If it's a folder, push subfolder's children URL
            if "folder" in item:
                sub_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{item['id']}/children?$top=200"
                stack.append(sub_url)
            results.append(item)

        if "@odata.nextLink" in data:
            stack.append(data["@odata.nextLink"])
    return results


def _list_drive_based(token: str, drive_id: str, folder_id: Optional[str] = None) -> List[dict]:
    """
    Drive-based approach:
      GET /drives/{driveId}/items/{folderId or root}/children
      Then handle subfolders the same way (stack).
    """
    if not folder_id:
        folder_id = "root"
    base_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children?$top=200"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    stack = [base_url]
    while stack:
        url = stack.pop()
        log.debug(f"[_list_drive_based] GET => {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("value", [])
        for item in items:
            if "folder" in item:
                sub_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item['id']}/children?$top=200"
                stack.append(sub_url)
            results.append(item)
        if "@odata.nextLink" in data:
            stack.append(data["@odata.nextLink"])
    return results


def _list_spo_list_items(token: str, site_id: str, list_id: str) -> List[dict]:
    """
    If you have a real SharePoint List, you can use the list endpoint:
      GET /sites/{siteId}/lists/{listId}/items?$expand=fields
    """
    base_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?$expand=fields"
    headers = {"Authorization": f"Bearer {token}"}

    results = []
    next_url = base_url
    while next_url:
        log.debug(f"[_list_spo_list_items] GET => {next_url}")
        resp = requests.get(next_url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("value", [])
        results.extend(items)
        next_url = data.get("@odata.nextLink")
    return results


def list_all_sharepoint_files_fallback(token: str, site_id: str, list_or_folder_id: str, is_list: bool) -> List[dict]:
    """
    Attempts to list either a SharePoint List or a SharePoint Drive folder recursively.
      - If is_list=True, calls _list_spo_list_items
      - Otherwise, tries site-based approach first
         if 400/404 => fallback to drive-based approach
    """
    log.debug("[list_all_sharepoint_files_fallback] Starting listing logic.")
    if is_list:
        log.info("Treating this as a real SharePoint List.")
        return _list_spo_list_items(token, site_id, list_or_folder_id)

    # Not a list => let's try site-based approach first
    try:
        log.info("Trying site-based approach for listing SharePoint folder items...")
        return _list_site_based(token, site_id, list_or_folder_id)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code in (400, 404):
            log.warning("Site-based approach failed. Attempting drive-based approach now.")
            return _list_drive_based(token, list_or_folder_id, None)
        else:
            raise e


def _download_site_based(token: str, site_id: str, item_id: str) -> bytes:
    """
    GET /sites/{siteId}/drive/items/{itemId}/content
    """
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    return resp.content


def _download_drive_based(token: str, drive_id: str, item_id: str) -> bytes:
    """
    GET /drives/{driveId}/items/{itemId}/content
    """
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    return resp.content


def download_file_with_fallback(token: str, site_id: str, item_id: str, maybe_drive_id: str) -> bytes:
    """
    Tries site-based download first. If that fails with 400/404 => fallback to drive-based approach.
    """
    try:
        log.info("[download_file_with_fallback] Trying site-based download first...")
        return _download_site_based(token, site_id, item_id)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code in (400, 404):
            log.warning("Site-based download failed => fallback to drive-based approach.")
            return _download_drive_based(token, maybe_drive_id, item_id)
        else:
            raise e
