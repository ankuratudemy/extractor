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

def validate_api_key(api_key):
    tenant_data = decode_api_key(api_key)
    if not tenant_data:
        return False

    # Extract necessary fields
    tenant_id = tenant_data.get('tenant_id')
    user_id = tenant_data.get('user_id')
    project_id = tenant_data.get('project_id')
    rate_limit = tenant_data.get('rate_limit')

    # Verify that the API key exists in Redis
    retries = 0
    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            key_exists = redis_conn.exists(api_key)
            if not key_exists:
                print("API key not found in Redis")
                return False

            # Optionally check for revocation or expiration here

            # Close Redis connection
            redis_conn.close()

            # Return the tenant_data for use in the request
            return tenant_data

        except redis.RedisError as e:
            print(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)

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
        decoded_data = decode_base64_url(encoded_data)
        calculated_signature = hashlib.sha256(f"{decoded_data}{SECRET_KEY}".encode()).hexdigest()

        if calculated_signature == signature:
            tenant_data = json.loads(decoded_data)
            return tenant_data
        else:
            print("Invalid signature")
            return False
    except Exception as e:
        print(f"Error decoding API key: {str(e)}")
        return False

def decode_base64_url(encoded_data):
    try:
        # Add padding if necessary
        padding_needed = 4 - (len(encoded_data) % 4)
        if padding_needed:
            encoded_data += "=" * padding_needed
        decoded_bytes = base64.urlsafe_b64decode(encoded_data)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        print(f"Error decoding base64 data: {str(e)}")
        return False
    
def api_key_required(token):
    api_key = token
    if not api_key:
        return False

    tenant_data = validate_api_key(api_key)
    if not tenant_data:
        return False

    # Attach tenant_data to the request for later use
    request.tenant_data = tenant_data
    return True
