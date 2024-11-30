import os
import sys
import json
import base64
import psycopg2
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


# Google Cloud Function entry point
def update_file_status(id, status, page_nums, updatedAt):
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                # Start the transaction
                connection.autocommit = False

                try:
                    print(f" id: {id} status: {status} page_nums {page_nums} udpatedAt {updatedAt}")
                    # Check if 'username' is provided, if not, fetch using 'userid'
                    if not id or not status:
                        raise ValueError("Either 'status', 'id', or page count is missing.")
                    # Acquire an advisory lock to serialize access to the credit update
                    lock_id = int(time.time() * 1000)  # milliseconds since the epoch
                    cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                    # Update CreditUsage table
                    cursor.execute(
                        'UPDATE "File" SET "status" = %s, "pageCount" = %s, "updatedAt" = %s WHERE "id" = %s',
                        (status, page_nums, updatedAt, id ),
                    )
                    connection.commit()
                    print("Transaction completed successfully.")
                    return "success"

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
