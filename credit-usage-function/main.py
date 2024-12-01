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


def update_remaining_credits(key, value):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            res = redis_conn.set(name=key, value=value)
            print(f"Redis db response: {res}")
            redis_conn.close()
            return True
        except redis.RedisError as e:
            print(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)
    print(f"Failed after {MAX_RETRIES} retries to update credits")
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

    user_id = data.get('user_id')
    project_id = data.get('project_id')
    credits_used = data.get('creditsUsed')

    # Perform the PostgreSQL transaction
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                # Start the transaction
                connection.autocommit = False

                try:
                    # Acquire an advisory lock to serialize access
                    # milliseconds since the epoch
                    lock_id = int(time.time() * 1000)
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                    # Get the subscriptionId associated with the projectId
                    cursor.execute(
                        'SELECT "subscriptionId" FROM "Project" WHERE id = %s',
                        (project_id,)
                    )
                    result = cursor.fetchone()
                    if result:
                        subscription_id = result[0]
                    else:
                        print(f"Project {project_id} not found.")
                        connection.rollback()
                        return

                    # Insert into CreditUsage table
                    cursor.execute(
                        'INSERT INTO "CreditUsage" ("creditsUsed", "timestamp", "subscriptionId", "projectId", "userId") VALUES (%s, NOW(), %s, %s, %s)',
                        (credits_used, subscription_id, project_id, user_id)
                    )

                    # Update creditsUsed in Project table
                    cursor.execute(
                        'UPDATE "Project" SET "creditsUsed" = "creditsUsed" + %s WHERE id = %s',
                        (credits_used, project_id)
                    )

                    # Update remainingCredits in Subscription table
                    cursor.execute(
                        'UPDATE "Subscription" SET "remainingCredits" = "remainingCredits" - %s WHERE id = %s RETURNING "remainingCredits"',
                        (credits_used, subscription_id)
                    )
                    updated_remaining_credits = cursor.fetchone()[0]
                    print(
                        f"Updated remainingCredits: {updated_remaining_credits}")

                    # Optionally update Redis cache
                    subscription_key = f"subscription_{subscription_id}_remainingCredits"
                    update_remaining_credits(
                        subscription_key, updated_remaining_credits)

                    # Update User's lastActiveTimestamp
                    cursor.execute(
                        'UPDATE "User" SET "lastActiveTimestamp" = NOW() WHERE id = %s',
                        (user_id,)
                    )

                    # Commit the transaction
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
