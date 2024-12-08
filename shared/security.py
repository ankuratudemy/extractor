# security.py

import redis
from flask import request, abort
from functools import wraps
import base64
import os
import json
import hashlib
import time
from shared.logging_config import log
MAX_RETRIES = 5

# Assuming you have a Redis connection details
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
SECRET_KEY = os.environ.get('SECRET_KEY')

def validate_api_key(api_key):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            subscription_id = redis_conn.get(api_key)
            credits_remaining = redis_conn.get(f"subscription_{subscription_id}_credits_remaining")
            log.info(f"Credits Remaining {credits_remaining}")
            if not credits_remaining:
                return False
            if float(credits_remaining) <= 0:
                print(f"No credits left")
                return False
            if subscription_id is not None:
                log.info(f"Subscription ID {subscription_id}")
                redis_conn.close()
                tenant_data = decode_api_key(api_key)
                return tenant_data
            return False
        except redis.RedisError as e:
            print(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)  # You may adjust the sleep duration between retries

    # If retries are exhausted or the API key is not found, return False
    print(f"Failed after {MAX_RETRIES} retries to validate API key")
    return False

def encode_api_key(key_data):
    json_data = json.dumps(key_data)
    signature = hashlib.sha256(f"{json_data}{SECRET_KEY}".encode()).hexdigest()
    encoded_data = base64.urlsafe_b64encode(json_data.encode()).decode() + '.' + signature
    return encoded_data

def decode_api_key(encoded_key):
    try:
        encoded_data, signature = encoded_key.rsplit('.', 1)
        decoded_data = decode(encoded_data)
        log.info(f"decoded_data {decoded_data}")
        calculated_signature = hashlib.sha256(f"{decoded_data}{SECRET_KEY}".encode()).hexdigest()

        if calculated_signature == signature:
            return json.loads(decoded_data)
        else:
            print("Invalid signature")
            return False

    except Exception as e:
        print(f"Error decoding API key: {str(e)}")
        return False

def decode(encoded_data):
    try:
        decoded_bytes = base64.urlsafe_b64decode(encoded_data + '=' * (4 - len(encoded_data) % 4))
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error decoding data: {str(e)}")
        return False
    
def api_key_required(token):
    api_key = token
    # api_key = request.headers.get('API-KEY')
    if not api_key:
        return False

    tenant_data = validate_api_key(api_key)
    print(f"tenant data is {tenant_data}")
    if tenant_data is False:
        print(f"tenant data is False")
        return False
    request.tenant_data = tenant_data
    return True
