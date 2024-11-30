#!/bin/bash

java -cp "tika-server-standard-${TIKA_VERSION}.jar:tika-extras/*" org.apache.tika.server.core.TikaServerCli -c "/app/tika-config.yml" -h "0.0.0.0"
