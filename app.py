import os
import io
from flask import Flask, request
from werkzeug.utils import secure_filename
from shared import pdf_processor
import json
import concurrent.futures
import multiprocessing
import time
import sys
import tempfile
import requests
import aiohttp
import asyncio
import threading

sys.path.append('../')
from shared.logging_config import log

app = Flask(__name__)

UPLOADS_FOLDER = '/app/uploads'
app.config['UPLOAD_FOLDER'] = UPLOADS_FOLDER


def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

@app.route('/health', methods=['GET'])
def health_check():
    return json.dumps({"status": "ok"})

@app.route('/extract', methods=['POST'])
def extract_text():
    uploaded_file = request.files['file']

    if uploaded_file:
        # Save the uploaded file to a temporary location
        filename = secure_filename(uploaded_file.filename)
        file_extension = os.path.splitext(filename)[1][1:].lower()
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if file_extension == 'pdf':
            # Read the PDF file directly from memory
            pdf_data = uploaded_file.read()

            # Measure the time taken to split the PDF
            start_time = time.time()
            pages = pdf_processor.split_pdf(pdf_data)
            split_time = time.time() - start_time
            log.info(f"Time taken to split the PDF: {split_time * 1000} ms")

        elif file_extension in ['ppt', 'pptx', 'docx', 'doc']:
            uploaded_file.save(temp_file_path)
            # Convert the file to PDF using LibreOffice
            pdf_data = convert_to_pdf(temp_file_path, file_extension)

            if pdf_data:
                # Measure the time taken to split the PDF
                start_time = time.time()
                pages = pdf_processor.split_pdf(pdf_data)
                split_time = time.time() - start_time
                log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
            else:
                log.error('Conversion to PDF failed')
                return 'Conversion to PDF failed.', 400
        else:
            log.error('Unsupported file format')
            return 'Unsupported file format.', 400

        # # Get the number of available CPUs
        # num_cpus = multiprocessing.cpu_count()
        # log.info(num_cpus)

        loop = get_event_loop()
        results = loop.run_until_complete(process_pages_async(pages))

        # Build the JSON output
        json_output = []
        for page_num, result in enumerate(results, start=1):
            page_obj = {
                'page': page_num,
                'text': result.strip()
            }
            json_output.append(page_obj)
        json_string = json.dumps(json_output, indent=4)
        log.info(f"Extraction successful for file: {filename}")

        # if os.path.exists(temp_file_path):
        #     os.remove(temp_file_path)
        return json_string
    else:
        log.error('No file uploaded')
        return 'No file uploaded.', 400

def convert_to_pdf(file_path, file_extension):
    import subprocess

    try:
        # Specify the output PDF file path
        pdf_file_path = os.path.splitext(file_path)[0] + '.pdf'

        # Convert the file to PDF using LibreOffice
        command = [
            'soffice',
            '--headless',
            '--convert-to',
            'pdf:writer_pdf_Export:{"SelectPdfVersion":{"type":"long","value":"17"}, "UseTaggedPDF": {"type":"boolean","value":"true"}}',
            '--outdir',
            os.path.dirname(file_path),
            file_path,
        ]
        subprocess.run(command, check=True)

        if os.path.exists(pdf_file_path):
            with open(pdf_file_path, 'rb') as f:
                pdf_data = f.read()

            # Delete the output PDF file
            os.remove(pdf_file_path)

            return pdf_data
        else:
            log.error('Conversion to PDF failed: Output file not found')
            return None
    except subprocess.CalledProcessError as e:
        log.error(f'Conversion to PDF failed: {str(e)}')
        return None
    except Exception as e:
        log.error(f'Error during PDF conversion: {str(e)}')
        return None

async def async_put_request(session, url, payload, headers):
    async with session.put(url, data=payload, headers=headers) as response:
        return await response.text()
    
async def process_pages_async(pages):
    url = "http://extractor-tika-server-service:9998/tika"
    headers = {'Accept': 'text/plain', 'Content-Type': 'application/pdf'}

    async with aiohttp.ClientSession() as session:
        tasks = [async_put_request(session, url, page, headers) for page in pages]
        results = await asyncio.gather(*tasks)

    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)