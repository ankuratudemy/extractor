import http from 'k6/http';
import { check, sleep } from 'k6';

const fileData = open('/C:/Users/ankur/Downloads/tmp_file.docx', 'b');
const boundary = 'boundary123'; // Replace with a unique boundary string

export let options = {
  vus: 2, // Number of virtual users (threads)
  duration: '10m', // Test duration
};

export default function () {
  let formData = `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="tmp_file.docx"\r\nContent-Type: application/octet-stream\r\n\r\n${fileData}\r\n--${boundary}--\r\n`;

  let params = {
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    }
  };

  let res = http.post(
    'https://extractor.livelydesert-b9447faa.centralus.azurecontainerapps.io/extract',
    formData,
    params
  );

  console.log(res)

  // Check for a successful response
  check(res, {
    'is status 200': (r) => r.status === 200,
  });

  // Sleep for a short duration before the next request
  sleep(1);
}
