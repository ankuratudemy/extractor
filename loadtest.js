import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 40,         // Number of virtual users
  duration: '30m',  // Test duration
};

const binFile = open('/C:/Users/ankur/Downloads/Format of NOC From Registered Office Owner.docx', 'b');

export default function () {
  const params = {
    timeout: '600s',
  };

  const data = {
    file: http.file(binFile, 'tmp_file.docx'),
  };

  let headers = {
    'Accept': 'text/plain',
    'API-KEY': 'eyJ0ZW5hbnRfaWQiOiJzdXBwb3J0QHN0cnVjdGh1Yi5pbyIsImtleU5hbWUiOiJLZXkxIiwiY3JlYXRlZEF0IjoiMjAyNC0wMi0xNVQxODo0NTo0My41NzBaIn0', // Replace with your actual access token
  };

  // Use batch function to make multiple requests concurrently
  let responses = http.batch([
    ['POST', 'https://stage.api.structhub.io/extract', data, { headers: headers }, params],
    // Add more requests as needed
  ]);

  // Check status for each response
  responses.forEach((res) => {
    check(res, {
      'status is 200': (r) => r.status === 200,
    });
  });

  // Adjust sleep time to achieve an average of 3 requests per second
  sleep(1 / 3); // 1/3 seconds sleep between requests
}
