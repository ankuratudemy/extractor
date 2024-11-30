import http from 'k6/http';
import { check } from 'k6';
import { sleep } from 'k6';


export let options = {
  vus: 100,// number of virtual users
  duration: '2m', // test duration
};

const binFile = open('/C:/Users/ankur/Downloads/Format of NOC From Registered Office Owner.docx', 'b');


export default function () {
  const params = {
     timeout: "600s"
  };

    const data = {
        file: http.file(binFile, 'tmp_file.docx'),
      };

  let headers = {
    'Accept': 'text/plain',
    'API-KEY': 'eyJ0ZW5hbnRfaWQiOiAic3RydWN0aHViYWRtaW4xIiwgInJhdGVfbGltaXQiOiAiMjAwMDAwMC9taW51dGUifQ==.7004df1440115c49bc34c5b13a11362b70d6927fa0e94d59af5cc01f5cf66342', // Replace with your actual access token
  };

  let res = http.post('https://stage.api.structhub.io/extract', data, { headers: headers }, params);
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  // Adjust sleep time if needed
  sleep(1);
}
