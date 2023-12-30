import os
import sys
import json
from PyPDF2 import PdfFileReader, PdfFileWriter
import concurrent.futures
import subprocess
import functools
import io
import time
sys.path.append('../')
from .logging_config import log

def split_page(page_num, pdf_data):
    pdf_reader = PdfFileReader(io.BytesIO(pdf_data))
    pdf_writer = PdfFileWriter()
    pdf_writer.addPage(pdf_reader.getPage(page_num))

    page_data = io.BytesIO()
    pdf_writer.write(page_data)
    page_data.seek(0)

    return page_data

def split_pdf(pdf_data):
    pages = []
    pdf_reader = PdfFileReader(io.BytesIO(pdf_data))
    num_pages = pdf_reader.getNumPages()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        page_indexes = range(num_pages)
        # Use functools.partial to create a partial function with pdf_data as a fixed argument
        split_page_partial = functools.partial(split_page, pdf_data=pdf_data)
        page_data_results = executor.map(split_page_partial, page_indexes)
        pages = list(page_data_results)
    
    return pages

def process_page(page_data):
    command = [
        'java',
        '-Xms2048m',
        '-Xmx2048m',
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
        extracted_text = stdout.decode('utf-8')

        end_time_page = time.time()  # Record the end time
        page_processing_time = end_time_page - start_time_page
        log.info(f"Time taken to process page: {page_processing_time * 1000} ms")

        return extracted_text
    except Exception as e:
        log.error(f"Error processing page. Error: {str(e)}")
        return ''
