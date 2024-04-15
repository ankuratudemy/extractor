import os
import sys
import json
import base64
import psycopg2
from google.api_core import retry
from google.cloud import pubsub_v1
import redis 
import time


# Retrieve PostgreSQL connection parameters from environment variables
db_params = {
    "host": os.environ.get("PSQL_HOST"),
    "port": os.environ.get("PSQL_PORT", "5432"),  # Default port is 5432
    "database": os.environ.get("PSQL_DATABASE"),
    "user": os.environ.get("PSQL_USERNAME"),
    "password": os.environ.get("PSQL_PASSWORD"),
}

MAX_RETRIES = 5

# Assuming you have a Redis connection details
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
SECRET_KEY = os.environ.get('SECRET_KEY')

def update_remaining_credits(username_credits_remaining, value):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            res = redis_conn.set(name=username_credits_remaining, value=value)
            print(f"Redis db response: {res}")
            return True
        except redis.RedisError as e:
            print(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)  # You may adjust the sleep duration between retries
            sys.exit("Function execution failed.")
    # If retries are exhausted or the API key is not found, return False
    print(f"Failed after {MAX_RETRIES} retries to validate API key")
    return False


# Google Cloud Function entry point
def pubsub_to_postgresql(event, context):
    # Decode the base64-encoded Pub/Sub message
    pubsub_message = event["data"]
    message_data = base64.b64decode(pubsub_message).decode("utf-8")

    # Parse the message data as JSON
    try:
        data = json.loads(message_data)
        print(f"Data Received from topic: {data}")
    except json.JSONDecodeError:
        print("Error decoding Pub/Sub message.")
        return

    # Perform the PostgreSQL transaction
    # Perform the PostgreSQL transaction
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                # Start the transaction
                connection.autocommit = False

                try:
                    # Acquire an advisory lock to serialize access to the credit update
                    lock_id = int(time.time() * 1000)  # milliseconds since the epoch
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                    # Update CreditUsage table
                    cursor.execute(
                        'INSERT INTO "CreditUsage" ("username", "creditsUsed") VALUES (%s, %s)',
                        (data["username"], data["creditsUsed"]),
                    )

                    # Update User table (subtract creditsUsed from credits)
                    cursor.execute(
                        'UPDATE "User" SET "credits" = "credits" - %s WHERE "username" = %s RETURNING credits',
                        (data["creditsUsed"], data["username"]),
                    )
                    # Fetch the updated credits value
                    updated_credits = cursor.fetchone()[0]
                    print(f"updated_credits: {updated_credits}")
                    remaining_credits_key = f"{data['username']}_credits_remaining"
                    # Update Redis key with the remaining credits value
                    update_remaining_credits(remaining_credits_key, updated_credits)
                    print(f"Updated Redis key {remaining_credits_key} with new value: {updated_credits}")
                    # Update User table (add creditsUsed to creditsUsed in User table)
                    cursor.execute(
                        'UPDATE "User" SET "creditsUsed" = "creditsUsed" + %s WHERE "username" = %s',
                        (data["creditsUsed"], data["username"]),
                    )
                    # Commit the transaction if both queries succeed
                    connection.commit()
                    print("Transaction completed successfully.")

                except Exception as e:
                    # Rollback the transaction if any part fails
                    connection.rollback()
                    print(f"Error during transaction: {str(e)}")
                    sys.exit("Function execution failed.")

    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        sys.exit("Function execution failed.")


# Sample usage:
# pubsub_to_postgresql({"data": "base64_encoded_message"}, None)
