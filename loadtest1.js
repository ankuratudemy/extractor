import http from 'k6/http';
import { check } from 'k6';
import { sleep } from 'k6';
let params = {
    timeout: '180s'
  };
export let options = {
  vus: 10, // number of virtual users
  duration: '10m', // test duration
};

const binFile = open('/C:/Users/ankur/Downloads/tmp_file.docx', 'b');


export default function () {
    const data = {
        file: http.file(binFile, 'tmp_file.docx'),
      };

  let headers = {
    'Accept': 'text/plain',
    'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImJkYzRlMTA5ODE1ZjQ2OTQ2MGU2M2QzNGNkNjg0MjE1MTQ4ZDdiNTkiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOiJzdGFnZS5hcGkuc3RydWN0aHViLmlvIiwiYXpwIjoieHRyYWN0LWZlLXNlcnZpY2UtYWNjb3VudEBzdHJ1Y3RodWItNDEyNjIwLmlhbS5nc2VydmljZWFjY291bnQuY29tIiwiZW1haWwiOiJ4dHJhY3QtZmUtc2VydmljZS1hY2NvdW50QHN0cnVjdGh1Yi00MTI2MjAuaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzA3MDMzMDEwLCJpYXQiOjE3MDcwMjk0MTAsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsInN1YiI6IjExNDg5MTQ0NzUyNzQwNzI0NjY4NiJ9.VX97d7eeOsUE2a2gxxSxn0PowmRGnUX4IGsaNrUTliQ3noD4H9qgPv8UQF6SGtKuDphIjMc8-sUX1aSRGXLHUMRO97xgR1mgvFwve-QR5z8r3n9bTJebvEInUdusZ0zX_N_5BhalOOUOdc6AZKv87SXmjBqtyQDXdlq2wFf_p9k8OJk2OhP3vsDb6ZnyOjfV3BJf8-FHyRN5LOtgvkrUHJEDwQGoeF_3Vs3Soa97TN2jfuy7cKMQBuANncWNbvWfG2EPQCeyXWsxXFJi8vvUgt8pjYeoey3mMzcCG3meGRLenwlNnfe6m83aEDuRigxWDyK7jihzXHsDzsm9vCxNSg', // Replace with your actual access token
  };

  let res = http.post('https://stage.api.structhub.io/extract', data, { headers: headers }, params);
  console.log(res.status)
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  // Adjust sleep time if needed
  sleep(1);
}
