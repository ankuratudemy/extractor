import os
import sys
import json
import base64
import psycopg2
import redis
import time
import uuid


# Retrieve PostgreSQL connection parameters from environment variables
db_params = {
    "host": os.environ.get("PSQL_HOST"),
    "port": os.environ.get("PSQL_PORT", "5432"),  # Default port is 5432
    "database": os.environ.get("PSQL_DATABASE"),
    "user": os.environ.get("PSQL_USERNAME"),
    "password": os.environ.get("PSQL_PASSWORD"),
}

MAX_RETRIES = 5

# Redis connection details
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = os.environ.get('REDIS_PORT')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')

def update_remaining_credits(key_name, value):
    retries = 0

    while retries < MAX_RETRIES:
        try:
            redis_conn = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            res = redis_conn.set(name=key_name, value=value)
            print(f"Redis db response: {res}")
            return True
        except redis.RedisError as e:
            print(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)  # You may adjust the sleep duration between retries
    print(f"Failed after {MAX_RETRIES} retries to update Redis key")
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

    # Extract data from the event
    subscription_id = data.get('subscription_id')
    user_id = data.get('user_id')
    project_id = data.get('project_id')
    credits_used = data.get('creditsUsed')

    if not subscription_id or not user_id or not project_id or not credits_used:
        print("Missing required data in the event.")
        return

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

                    # Insert into CreditUsage table
                    cursor.execute(
                        'INSERT INTO "CreditUsage" ("id","creditsUsed", "timestamp", "subscriptionId", "projectId", "userId") VALUES (%s, %s, NOW(), %s, %s, %s)',
                        (str(uuid.uuid4()),credits_used, subscription_id, project_id, user_id)
                    )

                    # Update Subscription table (subtract creditsUsed from remainingCredits)
                    cursor.execute(
                        'UPDATE "Subscription" SET "remainingCredits" = "remainingCredits" - %s WHERE "id" = %s RETURNING "remainingCredits"',
                        (credits_used, subscription_id)
                    )
                    subscription_remaining_credits = cursor.fetchone()
                    if subscription_remaining_credits:
                        subscription_remaining_credits = subscription_remaining_credits[0]
                        print(f"Subscription remaining credits: {subscription_remaining_credits}")
                    else:
                        raise Exception(f"Subscription ID {subscription_id} not found.")

                    # Update Project table (add creditsUsed to creditsUsed)
                    cursor.execute(
                        'UPDATE "Project" SET "creditsUsed" = "creditsUsed" + %s WHERE "id" = %s RETURNING "creditsUsed"',
                        (credits_used, project_id)
                    )
                    project_credits_used = cursor.fetchone()
                    if project_credits_used:
                        project_credits_used = project_credits_used[0]
                        print(f"Project credits used: {project_credits_used}")
                    else:
                        raise Exception(f"Project ID {project_id} not found.")

                    # Optional: Update User table (e.g., update lastActiveTimestamp)
                    cursor.execute(
                        'UPDATE "User" SET "lastActiveTimestamp" = NOW() WHERE "id" = %s',
                        (user_id,)
                    )

                    # Commit the transaction if all queries succeed
                    connection.commit()
                    print("Transaction completed successfully.")

                    # Update Redis keys with the remaining credits values
                    subscription_credits_key = f"subscription_{subscription_id}_remaining_credits"
                    update_remaining_credits(subscription_credits_key, subscription_remaining_credits)
                    print(f"Updated Redis key {subscription_credits_key} with new value: {subscription_remaining_credits}")

                except Exception as e:
                    # Rollback the transaction if any part fails
                    connection.rollback()
                    print(f"Error during transaction: {str(e)}")
                    sys.exit("Function execution failed.")

    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        sys.exit("Function execution failed.")
