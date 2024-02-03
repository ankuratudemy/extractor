import http from 'k6/http';
import { check, sleep } from 'k6';

const fileData = open('/C:/Users/ankur/Downloads/tmp_file.docx', 'b');
const boundary = 'boundary123'; // Replace with a unique boundary string

export let options = {
  vus: 2, // Number of virtual users (threads)
  duration: '10s', // Test duration
};

export default function () {
  let formData = `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="tmp_file.docx"\r\nContent-Type: application/octet-stream\r\n\r\n${fileData}\r\n--${boundary}--\r\n`;

  let params = {
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Authorization': `Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Ijg1ZTU1MTA3NDY2YjdlMjk4MzYxOTljNThjNzU4MWY1YjkyM2JlNDQiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJzdGFnZS5hcGkuc3RydWN0aHViLmlvIiwiYXpwIjoieHRyYWN0LWZlLXNlcnZpY2UtYWNjb3VudEBzdHJ1Y3RodWItNDEyNjIwLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwiZW1haWwiOiJ4dHJhY3QtZmUtc2VydmljZS1hY2NvdW50QHN0cnVjdGh1Yi00MTI2MjAuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzA3MDAzMzc3LCJpYXQiOjE3MDY5OTk3NzcsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsInN1YiI6IjExNDg5MTQ0NzUyNzQwNzI0NjY4NiJ9.D8QyTdr5Kx-4WHypZ_zLI1tNDhVrADEFAEowvxb3WMhGpqKqoj_4z7IRJAkneas4YIEA17VcE7EzA0-kFzakAwS-t5m8w4anGyGC2ZkKkrbZTrLt9xqu16UG7t1FX1-OFSitLQDtA2TtmRZnVvUUW_YWV1bCwa_-Fyv11kIuUIm86LO8VtP90Z18-qFUpRaBSjyx3bz-4PlWMv9l1YfCiTmQdR8ZZ9CNT2ne1F3JMU1rzWrB4K8cd6wVtgUgd4-s9AU2cTibC3OE7TxmBGOaenypHfaZbF6naIXwwKPk7sGC9G9-nnMv-XLwuEdGkLGdsgSOyCEShwtTICfLX5L-5g`
    }
  };

  let res = http.post(
    'https://stage.api.structhub.io/extract',
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
