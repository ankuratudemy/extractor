import http from 'k6/http';
import { check } from 'k6';
import { sleep } from 'k6';


export let options = {
  vus: 100, // number of virtual users
  duration: '10m', // test duration
};

const binFile = open('/C:/Users/ankur/Downloads/tmp_file.docx', 'b');


export default function () {
  const params = {
     timeout: '6000s'
  };

    const data = {
        file: http.file(binFile, 'tmp_file.docx'),
      };

  let headers = {
    'Accept': 'text/plain',
    'API-KEY': 'eyJ0ZW5hbnRfaWQiOiAic3RydWN0aHViYWRtaW4xIiwgInJhdGVfbGltaXQiOiAiMjAvbWludXRlIn0=.c083eba28285fed449eb159494faaa2e2e9714bbfbb053ee4abc8d0966fe9319', // Replace with your actual access token
  };

  let res = http.post('https://stage.api.structhub.io/extract', data, { headers: headers }, params);
  console.log(res.status)
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  // Adjust sleep time if needed
  sleep(1);
}
