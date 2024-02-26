import os
import sys
import json
import fitz  # PyMuPDF library
import concurrent.futures
import pandas as pd
import subprocess
import pyexcel as pe
import functools
import io
from io import BytesIO
import pandas as pd
import time
sys.path.append('../')
from .logging_config import log

# def split_page(page_num, pdf_data):
#     pdf_reader = PdfFileReader(io.BytesIO(pdf_data))
#     pdf_writer = PdfFileWriter()
#     pdf_writer.addPage(pdf_reader.getPage(page_num))

#     page_data = io.BytesIO()
#     pdf_writer.write(page_data)
#     page_data.seek(0)

#     return page_data

# def split_pdf(pdf_data):
#     pages = []
#     pdf_reader = PdfFileReader(io.BytesIO(pdf_data))
#     num_pages = pdf_reader.getNumPages()
#     log.info(f'Number of pages: {num_pages}')
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         page_indexes = range(num_pages)
#         # Use functools.partial to create a partial function with pdf_data as a fixed argument
#         split_page_partial = functools.partial(split_page, pdf_data=pdf_data)
#         page_data_results = executor.map(split_page_partial, page_indexes)
#         pages = list(page_data_results)
    
#     return pages
def split_excel(excel_bytes):
    # Read the Excel bytes data into a dictionary of DataFrames
    dfs = pd.read_excel(BytesIO(excel_bytes), sheet_name=None, engine='openpyxl')

    # Create a list of tuples with sheet names and corresponding data as BytesIO
    result = [(name, BytesIO()) for name in dfs]
    print(result)
    # Write each DataFrame to the corresponding BytesIO object
    for name, bio in result:
        dfs[name].to_excel(bio, index=False, engine='openpyxl')
        bio.seek(0)  # Reset the position to the beginning of the BytesIO object

    return result


def split_ods(ods_bytes):
    # Create a dictionary to store sheet name and corresponding BytesIO data
    result = []

    # Read the ODS file using pandas
    try:
        xls = pd.ExcelFile(BytesIO(ods_bytes), engine="odf")
    except Exception as e:
        raise ValueError("Invalid ODS file format. Make sure it's a valid spreadsheet in ODS format.") from e

    # Iterate through sheets and store each sheet's name and BytesIO data in a tuple
    for sheet_name in xls.sheet_names:
        sheet_data = xls.parse(sheet_name)

        # Convert the DataFrame to BytesIO
        sheet_bytes = BytesIO()
        sheet_data.to_excel(sheet_bytes, index=False, engine="odf")
        sheet_bytes.seek(0)

        result.append((sheet_name, sheet_bytes))

    return result


def split_page(page_num, pdf_data):
    pdf_document = fitz.open(stream=pdf_data)
    pdf_page = pdf_document[page_num]
    pdf_writer = fitz.open()

    pdf_writer.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
    
    page_data = io.BytesIO()
    pdf_writer.save(page_data)
    pdf_writer.close()
    
    page_data.seek(0)
    pdf_document.close()

    return page_num+1, page_data  # Return both page number and page data

def split_pdf(pdf_data):
    pages = []
    num_pages = fitz.open(stream=pdf_data).page_count
    print(f'Number of pages: {num_pages}')

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        page_indexes = range(num_pages)
        split_page_partial = functools.partial(split_page, pdf_data=pdf_data)
        page_data_results = executor.map(split_page_partial, page_indexes)
        pages = list(page_data_results)

    return pages, num_pages

def process_page(page_data):
    command = [
        'java',
        '-Xms1024m',
        '-Xmx1024m',
        '-jar',
        './tika-app.jar',
        '-t',
        '-'
    ]
    try:
        start_time_page = time.time()  # Record the start time

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(input=page_data.read())
        log.error(stderr)
        extracted_text = stdout.decode('utf-8')

        end_time_page = time.time()  # Record the end time
        page_processing_time = end_time_page - start_time_page
        log.info(f"Time taken to process page: {page_processing_time * 1000} ms")

        return extracted_text
    except Exception as e:
        log.error(f"Error processing page. Error: {str(e)}")
        return ''
