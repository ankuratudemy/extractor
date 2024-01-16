import os
import sys
import json
import fitz  # PyMuPDF library
import concurrent.futures
import subprocess
import functools
import io
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

    return pages

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
