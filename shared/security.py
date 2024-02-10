# security.py

import redis
from flask import request, abort
from functools import wraps
import base64
import os
import json
import hashlib
import time

MAX_RETRIES = 5

# Assuming you have a Redis connection details
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
SECRET_KEY = os.environ.get('SECRET_KEY')

def create_api_key(tenant_id, key_data):
    encoded_key = encode_api_key(key_data)
    redis_conn = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
    redis_conn.set(encoded_key, tenant_id)

def validate_api_key(api_key):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            tenant_id = redis_conn.get(api_key)

            if tenant_id is not None:
                redis_conn.close()
                tenant_data = decode_api_key(api_key)
                return tenant_data

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

def decode_api_key(api_key):
    try:
        encoded_data, signature = api_key.split('.')
        decoded_data = base64.urlsafe_b64decode(encoded_data.encode()).decode()
        calculated_signature = hashlib.sha256(f"{decoded_data}{SECRET_KEY}".encode()).hexdigest()

        if calculated_signature != signature:
            return False

        return json.loads(decoded_data)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False

def api_key_required(token):
    api_key = token
    # api_key = request.headers.get('API-KEY')
    if not api_key:
        return False

    tenant_data = validate_api_key(api_key)
    request.tenant_data = tenant_data
    return True