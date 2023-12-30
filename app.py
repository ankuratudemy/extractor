from flask import Flask, request
from shared import pdf_processor
import json
import concurrent.futures
import multiprocessing
import time
import sys
sys.path.append('../')
from shared.logging_config import log

app = Flask(__name__)
# app.debug = os.environ.get('FLASK_ENV') == 'development'


@app.route('/extract', methods=['POST'])
def extract_text():
    file = request.files['file']
    # Check if the uploaded file is a PDF
    if file.content_type == 'application/pdf':
        # Read the PDF file directly from memory
        pdf_data = file.read()

        # Measure the time taken to split the PDF
        start_time = time.time()
        pages = pdf_processor.split_pdf(pdf_data)
        split_time = time.time() - start_time
        log.info(f"Time taken to split the PDF: {split_time * 1000} ms")

        # Get the number of available CPUs
        num_cpus = multiprocessing.cpu_count()
        log.info(num_cpus)
        # Process the pages concurrently using multiple processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = executor.map(pdf_processor.process_page, pages)

        # Build the JSON output
        json_output = []
        for page_num, result in enumerate(results, start=1):
            page_obj = {
                'page': page_num,
                'text': result.strip()
            }
            json_output.append(page_obj)
        json_string = json.dumps(json_output, indent=4)

        log.info(f"Extraction successful for file: {file.filename}")

        return json_string
    else:
        log.error(f"Uploaded file is not a PDF: {file.filename}")
        return 'Uploaded file is not a PDF.', 400
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)