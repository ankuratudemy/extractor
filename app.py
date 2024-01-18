import os
import io
from flask import Flask, request
from werkzeug.utils import secure_filename
from shared import file_processor
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

SERVER_URL = f"http://{os.environ.get('SERVER_URL')}/tika"
# TIKA_FUNCTION = "http://host.docker.internal:9998/tika"

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
    # # Access headers
    # all_headers = request.headers

    # # Print all headers
    # print("All Headers:")
    # for key, value in all_headers.items():
    #     print(f"{key}: {value}")

    # # Access a specific header
    # content_type_header = request.headers.get('Content-Type')
    # print(f"\nContent-Type Header: {content_type_header}")

    contentType='application/pdf'
    uploaded_file = request.files['file']

    if uploaded_file:
        # Save the uploaded file to a temporary location
        filename = secure_filename(uploaded_file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(uploaded_file.content_type)
        
        oFileExtMap = {
            "application/octet-stream": "use_extension", # use file extension if content type is octet-stream
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/pdf": "pdf",
            "application/vnd.oasis.opendocument.text": "odt",
            "application/vnd.oasis.opendocument.spreadsheet": "ods",
            "application/vnd.oasis.opendocument.presentation": "odp",
            "application/vnd.oasis.opendocument.graphics": "odg",
            "application/vnd.oasis.opendocument.formula": "odf",
            "application/vnd.oasis.opendocument.flat.text": "fodt",
            "application/vnd.oasis.opendocument.flat.presentation": "fodp",
            "application/vnd.oasis.opendocument.flat.graphics": "fodg",
            "application/vnd.oasis.opendocument.spreadsheet-template": "ots",
            "application/vnd.oasis.opendocument.flat.spreadsheet-template": "fots",
            "application/vnd.lotus-1-2-3": "123",
            "application/dbase": "dbf",
            "text/html": "html",
            "application/vnd.lotus-screencam": "scm",
            "text/csv": "csv",
            "application/vnd.ms-excel": "xls",
            "application/vnd.ms-excel.template.macroenabled.12": "xltm",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.template": 'dotx',
            "application/vnd.ms-word.document.macroenabled.12": 'docm',
            "application/vnd.ms-word.template.macroenabled.12": 'dotm',
            "application/xml": 'xml',
            "application/msword": 'doc',
            "application/vnd.ms-word.document.macroenabled.12": 'docm',
            "application/vnd.ms-word.template.macroenabled.12": 'dotm',
            "application/rtf": 'rtf',
            "application/wordperfect": 'wpd',
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 'xlsx',
            "application/vnd.openxmlformats-officedocument.spreadsheetml.template": 'xltx',
            "application/vnd.ms-excel.sheet.macroenabled.12": 'xlsm',
            "application/vnd.ms-excel.template.macroenabled.12": 'xltm',
            "application/vnd.corelqpw": 'qpw',
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": 'pptx',
            "application/vnd.openxmlformats-officedocument.presentationml.slideshow": 'ppsx',
            "application/vnd.openxmlformats-officedocument.presentationml.slide": 'ppmx',
            "application/vnd.openxmlformats-officedocument.presentationml.template": 'potx',
            "application/vnd.ms-powerpoint": 'ppt',
            "application/vnd.ms-powerpoint.slideshow.macroenabled.12": 'ppsm',
            "application/vnd.ms-powerpoint.presentation.macroenabled.12": 'pptm',
            "application/vnd.ms-powerpoint.addin.macroenabled.12": 'ppam',
            "application/vnd.ms-powerpoint.slideshow.macroenabled.12": 'ppsm',
            "application/vnd.ms-powerpoint.presentation.macroenabled.12": 'pptm',
            "application/vnd.ms-powerpoint.addin.macroenabled.12": 'ppam',
            "application/vnd.ms-powerpoint": 'ppt',
            "application/vnd.ms-powerpoint.slideshow": 'pps',
            "application/vnd.ms-powerpoint.presentation": 'ppt',
            "application/vnd.ms-powerpoint.addin": 'ppa',
        }
        # Reverse mapping of content types to file extensions
        reverse_file_ext_map = {v: k for k, v in oFileExtMap.items()}

        if uploaded_file.content_type not in oFileExtMap:
            print('Invalid file extension')

        file_extension = oFileExtMap[ uploaded_file.content_type ]
        
        if file_extension == 'use_extension':
            file_extension = os.path.splitext(filename)[1][1:].lower()

        print(file_extension)
        print(filename)

        if file_extension == 'pdf':
            # Read the PDF file directly from memory
            pdf_data = uploaded_file.read()

            # Measure the time taken to split the PDF
            start_time = time.time()
            pages = file_processor.split_pdf(pdf_data)
            split_time = time.time() - start_time
            log.info(f"Time taken to split the PDF: {split_time * 1000} ms")
        
        elif file_extension in ['csv', 'xls', 'xltm', 'xltx', 'xlsx', 'tsv', 'ots']:
            pages = file_processor.split_excel(uploaded_file.read())
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['ods']:
            print("here")
            pages = file_processor.split_ods(uploaded_file.read())
            print(pages)
            contentType = reverse_file_ext_map.get(file_extension, '')

        elif file_extension in ['docx', 'pdf', 'odt', 'odp', 'odg', 'odf', 'fodt', 'fodp', 'fodg', '123', 'dbf', 'html', 'scm', 'dotx', 'docm', 'dotm', 'xml', 'doc',  'qpw', 'pptx', 'ppsx', 'ppmx', 'potx', 'pptm', 'ppam', 'ppsm', 'pptm', 'ppam', 'ppt', 'pps', 'ppt', 'ppa', 'rtf']:
            uploaded_file.save(temp_file_path)
            # Convert the file to PDF using LibreOffice
            pdf_data = convert_to_pdf(temp_file_path, file_extension)

            if pdf_data:
                # Measure the time taken to split the PDF
                start_time = time.time()
                pages = file_processor.split_pdf(pdf_data)
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
        results = loop.run_until_complete(process_pages_async(pages, contentType))

        # Build the JSON output using mapped_results
        json_output = []
        for result, page_num in results:
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
            '/opt/libreoffice7.6/program/soffice',
            '--headless',
            '--nodefault',
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

async def async_put_request(session, url, payload, page_num, headers):
    async with session.put(url, data=payload, headers=headers) as response:
        return await response.text(), page_num

async def process_pages_async(pages, contentType):
    url = SERVER_URL
    headers = {'Accept': 'text/plain', 'Content-Type': contentType}

    async with aiohttp.ClientSession() as session:
        tasks = [async_put_request(session, url, page_data, page_num, headers) for page_num, page_data in pages]
        results = await asyncio.gather(*tasks)

    # Map results to their corresponding page numbers
    mapped_results = [(result, page_num) for (result, page_num) in results]

    return mapped_results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)