import os
import sys
import json
import base64
import psycopg2
import redis
import time
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
            redis_conn = redis.StrictRedis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            res = redis_conn.set(name=key_name, value=value)
            log.info(f"Redis db response for {key_name}: {res}")
            return True
        except redis.RedisError as e:
            log.error(f"Error connecting to Redis: {str(e)}")
            retries += 1
            time.sleep(1)  # You may adjust the sleep duration between retries
    log.error(f"Failed after {MAX_RETRIES} retries to update Redis key {key_name}")
    return False

# Google Cloud Function entry point
def pubsub_to_postgresql(event, context):
    # Decode the base64-encoded Pub/Sub message
    pubsub_message = event.get("data", "")
    if not pubsub_message:
        log.error("No data found in the Pub/Sub message.")
        return

    try:
        message_data = base64.b64decode(pubsub_message).decode("utf-8")
    except Exception as e:
        log.error(f"Error decoding Pub/Sub message: {str(e)}")
        return

    # Parse the message data as JSON
    try:
        data = json.loads(message_data)
        log.info(f"Data Received from topic: {data}")
    except json.JSONDecodeError:
        log.error("Error decoding Pub/Sub message as JSON.")
        return

    # Extract data from the event
    subscription_id = data.get('subscription_id')
    user_id = data.get('user_id')
    data_source_id = data.get('data_source_id')
    project_id = data.get('project_id')
    credits_used = data.get('creditsUsed')

    # Validate required fields
    if not subscription_id or not project_id or not credits_used or (not data_source_id and not user_id):
        log.error("Missing required data in the event.")
        return

    try:
        with psycopg2.connect(**db_params) as connection:
            connection.autocommit = False  # Start a transaction
            with connection.cursor() as cursor:
                # Use a consistent lock based on subscription_id to prevent race conditions
                lock_id = hash(subscription_id)  # Consistent lock ID
                try:
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))
                    log.info(f"Acquired advisory lock for subscription_id {subscription_id}.")
                except Exception as e:
                    log.error(f"Error acquiring advisory lock: {str(e)}")
                    return

                # Initialize variables to capture remaining credits
                subscription_remaining_credits = None

                # 1. Insert into CreditUsage table
                try:
                    cursor.execute(
                        'INSERT INTO "CreditUsage" ("id","creditsUsed", "timestamp", "subscriptionId", "projectId", "userId", "dataSourceId") VALUES (%s, %s, NOW(), %s, %s, %s, %s)',
                        (str(uuid.uuid4()), credits_used, subscription_id, project_id, user_id, data_source_id)
                    )
                    log.info("Inserted into CreditUsage table.")
                except psycopg2.errors.ForeignKeyViolation as e:
                    # Rollback to the start of the transaction to avoid affecting other operations
                    connection.rollback()
                    log.warning(f"ForeignKeyViolation while inserting CreditUsage: {str(e)}. Continuing without inserting CreditUsage.")
                    # Re-establish the transaction for further operations
                    connection.autocommit = False
                    cursor = connection.cursor()
                except Exception as e:
                    connection.rollback()
                    log.error(f"Error inserting into CreditUsage: {str(e)}. Continuing without inserting CreditUsage.")
                    # Re-establish the transaction for further operations
                    connection.autocommit = False
                    cursor = connection.cursor()

                # 2. Update Subscription table (subtract creditsUsed)
                try:
                    cursor.execute(
                        'UPDATE "Subscription" SET "remainingCredits" = "remainingCredits" - %s WHERE "id" = %s RETURNING "remainingCredits"',
                        (credits_used, subscription_id)
                    )
                    result = cursor.fetchone()
                    if result:
                        subscription_remaining_credits = result[0]
                        log.info(f"Subscription {subscription_id} remaining credits: {subscription_remaining_credits}")
                    else:
                        raise Exception(f"Subscription ID {subscription_id} not found.")
                except Exception as e:
                    log.error(f"Error updating Subscription table: {str(e)}. Continuing with other updates.")
                    # Continue without raising the exception

                # 3. Update Project table (add creditsUsed)
                try:
                    cursor.execute(
                        'UPDATE "Project" SET "creditsUsed" = "creditsUsed" + %s WHERE "id" = %s RETURNING "creditsUsed"',
                        (credits_used, project_id)
                    )
                    result = cursor.fetchone()
                    if result:
                        project_credits_used = result[0]
                        log.info(f"Project {project_id} credits used: {project_credits_used}")
                    else:
                        raise Exception(f"Project ID {project_id} not found.")
                except Exception as e:
                    log.error(f"Error updating Project table: {str(e)}. Continuing with other updates.")
                    # Continue without raising the exception

                # 4. Optionally update User table (e.g., update lastActiveTimestamp)
                if user_id:
                    try:
                        cursor.execute(
                            'UPDATE "User" SET "lastActiveTimestamp" = NOW() WHERE "id" = %s',
                            (user_id,)
                        )
                        log.info(f"Updated User {user_id} lastActiveTimestamp.")
                    except Exception as e:
                        log.error(f"Error updating User table: {str(e)}. Continuing with other updates.")
                        # Continue without raising the exception

                # Commit the transaction
                try:
                    connection.commit()
                    log.info("Transaction committed successfully.")
                except Exception as e:
                    connection.rollback()
                    log.error(f"Error committing transaction: {str(e)}")
                    return

                # 5. Update Redis with the remaining credits
                if subscription_remaining_credits is not None:
                    subscription_credits_key = f"subscription_{subscription_id}_remaining_credits"
                    if update_remaining_credits(subscription_credits_key, subscription_remaining_credits):
                        log.info(f"Updated Redis key {subscription_credits_key} with new value: {subscription_remaining_credits}")
                    else:
                        log.error(f"Failed to update Redis key {subscription_credits_key}")
                else:
                    log.warning(f"Subscription remaining credits not available. Redis update skipped.")

    except psycopg2.OperationalError as e:
        log.error(f"Error connecting to PostgreSQL: {str(e)}")
    except Exception as e:
        log.error(f"Unexpected error: {str(e)}")
