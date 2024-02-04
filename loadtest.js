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
console.log(formData)
  let params = {
    headers: {
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
      'Authorization': `Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImJkYzRlMTA5ODE1ZjQ2OTQ2MGU2M2QzNGNkNjg0MjE1MTQ4ZDdiNTkiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJzdGFnZS5hcGkuc3RydWN0aHViLmlvIiwiYXpwIjoieHRyYWN0LWZlLXNlcnZpY2UtYWNjb3VudEBzdHJ1Y3RodWItNDEyNjIwLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwiZW1haWwiOiJ4dHJhY3QtZmUtc2VydmljZS1hY2NvdW50QHN0cnVjdGh1Yi00MTI2MjAuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzA3MDEwMzkzLCJpYXQiOjE3MDcwMDY3OTMsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsInN1YiI6IjExNDg5MTQ0NzUyNzQwNzI0NjY4NiJ9.fkKBaaTdJGIHkl9XuncIOxtCiz9LmANy0qI1uFnKvivbqPdzOxUeaET8z80s0nCtvZbtJ2kvi8V4b_yQDdGk89g8SyE2K_ZmaxLISXjOUnLmEzTR-lalOdGPuUkNwNw9reJBZLAIfTP5y7jtTWmA7IQHxQ6_hLtZq93Nee8kAGNVijIE5VdCtJTJhKHbBh-UmrojB48MTuniVAHLAnh8-CXn76MELvB-CPNuezJtLOLe25GAzc08ZxOnMEwoP3Q0wtWpWLosDiTWlRDBM3acJ3VsZ5CedZ_nXWv9GmNChkZIToiNvjSl_yNjui9w3cSVqJAcKWuTKNjxNn3YCMz3Jg`
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
