import sys
import json
import fitz  # PyMuPDF
import pandas as pd
import functools
import io
from io import BytesIO
import time
sys.path.append('../')
from .logging_config import log

def split_excel(excel_bytes):
    try:
        dfs = pd.read_excel(BytesIO(excel_bytes), sheet_name=None, engine='openpyxl')
        result = []
        for name, df in dfs.items():
            bio = BytesIO()
            df.to_excel(bio, index=False, engine='openpyxl')
            bio.seek(0)
            result.append((name, bio))
            log.info(f"Split Excel sheet: {name}")
        return result
    except Exception as e:
        log.error(f"Error splitting Excel file: {str(e)}")
        raise

def split_ods(ods_bytes):
    result = []
    try:
        xls = pd.ExcelFile(BytesIO(ods_bytes), engine="odf")
    except Exception as e:
        log.error(f"Error opening ODS file: {str(e)}")
        raise ValueError("Invalid ODS file format.") from e

    for sheet_name in xls.sheet_names:
        try:
            sheet_data = xls.parse(sheet_name)
            sheet_bytes = BytesIO()
            sheet_data.to_excel(sheet_bytes, index=False, engine="openpyxl")
            sheet_bytes.seek(0)
            result.append((sheet_name, sheet_bytes))
            log.info(f"Split ODS sheet: {sheet_name}")
        except Exception as e:
            log.error(f"Error splitting ODS sheet {sheet_name}: {str(e)}")
            continue  # Skip problematic sheets

    return result

def split_page(pdf_document, page_num):
    try:
        pdf_writer = fitz.open()
        pdf_writer.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
        page_data = BytesIO()
        pdf_writer.save(page_data)
        pdf_writer.close()
        page_data.seek(0)
        log.info(f"Extracted page {page_num + 1}")
        return (page_num + 1, page_data)
    except Exception as e:
        log.error(f"Error extracting page {page_num + 1}: {str(e)}")
        raise

def split_pdf(pdf_data):
    try:
        pdf_document = fitz.open(stream=pdf_data)
        num_pages = pdf_document.page_count
        log.info(f"Splitting PDF with {num_pages} pages")
    except Exception as e:
        log.error(f"Error opening PDF document: {str(e)}")
        raise

    pages = []
    for page_num in range(num_pages):
        try:
            pg_num, p_data = split_page(pdf_document, page_num)
            pages.append((pg_num, p_data))
        except Exception as e:
            log.error(f"Skipping page {page_num + 1} due to error: {str(e)}")
            continue  # Skip problematic pages

    pdf_document.close()
    log.info(f"Successfully split PDF into {len(pages)} pages out of {num_pages}")
    return pages, len(pages)
