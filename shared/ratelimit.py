# ratelimit.py

from flask_limiter import Limiter
from flask import request, current_app, abort, jsonify
from flask_limiter.errors import RateLimitExceeded
import os


def check_rate_limit(limiter):
    tenant_data = getattr(request, 'tenant_data', None)
    print(tenant_data)
    if not tenant_data:
        # If tenant_data is not available, it means the API key check failed
        abort(401, "Invalid API Key")

    # Use tenant_data to get rate limit info and customize rate limits dynamically
    rate_limit_info = tenant_data.get('rate_limit')
    print(rate_limit_info)
    # Skip rate limiting for /health route
    if request.endpoint == 'health_check':
        return
    try:
        print(rate_limit_info)
        print(limiter)
        # Use a dynamic key_func here
        limiter.limit(limit_value=rate_limit_info)(current_app)
    except RateLimitExceeded as e:
        # Custom response for rate limit exceeded
        response = jsonify({"error": "Rate limit exceeded", "message": str(e.description)})
        response.status_code = 429  # HTTP 429 Too Many Requests
        return response
