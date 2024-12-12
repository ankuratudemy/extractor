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
    dfs = pd.read_excel(BytesIO(excel_bytes), sheet_name=None, engine='openpyxl')
    result = [(name, BytesIO()) for name in dfs]
    for name, bio in result:
        dfs[name].to_excel(bio, index=False, engine='openpyxl')
        bio.seek(0)
    return result

def split_ods(ods_bytes):
    result = []
    try:
        xls = pd.ExcelFile(BytesIO(ods_bytes), engine="odf")
    except Exception as e:
        raise ValueError("Invalid ODS file format.") from e

    for sheet_name in xls.sheet_names:
        sheet_data = xls.parse(sheet_name)
        sheet_bytes = BytesIO()
        sheet_data.to_excel(sheet_bytes, index=False, engine="openpyxl")
        sheet_bytes.seek(0)
        result.append((sheet_name, sheet_bytes))

    return result

def split_page(page_num, pdf_data):
    pdf_document = fitz.open(stream=pdf_data)
    pdf_writer = fitz.open()
    pdf_writer.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    page_data = io.BytesIO()
    pdf_writer.save(page_data)
    pdf_writer.close()
    pdf_document.close()
    page_data.seek(0)
    return page_num+1, page_data

def split_pdf(pdf_data):
    pdf_document = fitz.open(stream=pdf_data)
    num_pages = pdf_document.page_count
    pdf_document.close()
    log.info(f"Splitting PDF with {num_pages} pages")

    pages = []
    for page_num in range(num_pages):
        pg_num, p_data = split_page(page_num, pdf_data)
        pages.append((pg_num, p_data))
        log.info(f"Extracted page {pg_num}/{num_pages}")

    return pages, num_pages
