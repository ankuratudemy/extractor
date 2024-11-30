import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 40,         // Number of virtual users
  duration: '2m',  // Test duration
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
    'API-KEY': 'eyJ0ZW5hbnRfaWQiOiJpdEBzdHJ1Y3RodWIuaW8iLCJrZXlOYW1lIjoiVGVzdEtleSIsImNyZWF0ZWRBdCI6IjIwMjQtMDItMjJUMjI6MTA6NTAuMzU3WiIsInJhdGVfbGltaXQiOiI1L21pbnV0ZSJ9.ec7c88ae7a9150dd97d392682d76adfb06ad5e725b0ee1e43acb4cd00083da17', // Replace with your actual access token
  };

  // Use batch function to make multiple requests concurrently
  let responses = http.batch([
    ['POST', 'https://stage.api.structhub.io/extract', data, { headers: headers }, params],
    // Add more requests as needed
  ]);

  // Check status for each response
  responses.forEach((res) => {
    console.log(res.status)
    check(res, {
      'status is 200': (r) => r.status === 200,
    });
  });

  // Adjust sleep time to achieve an average of 3 requests per second
  sleep(1 / 3); // 1/3 seconds sleep between requests
}
