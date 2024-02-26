import os
import sys
import json
import base64
import psycopg2
from google.api_core import retry
from google.cloud import pubsub_v1

# Retrieve PostgreSQL connection parameters from environment variables
db_params = {
    "host": os.environ.get("PSQL_HOST"),
    "port": os.environ.get("PSQL_PORT", "5432"),  # Default port is 5432
    "database": os.environ.get("PSQL_DATABASE"),
    "user": os.environ.get("PSQL_USERNAME"),
    "password": os.environ.get("PSQL_PASSWORD"),
}


# Google Cloud Function entry point
def pubsub_to_postgresql(event, context):
    # Decode the base64-encoded Pub/Sub message
    pubsub_message = event["data"]
    message_data = base64.b64decode(pubsub_message).decode("utf-8")

    # Parse the message data as JSON
    try:
        data = json.loads(message_data)
    except json.JSONDecodeError:
        print("Error decoding Pub/Sub message.")
        return

    # Perform the PostgreSQL transaction
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                # Start the transaction
                connection.autocommit = False

                try:
                    # Update CreditUsage table
                    cursor.execute(
                        'INSERT INTO "CreditUsage" ("username", "creditsUsed") VALUES (%s, %s)',
                        (data["username"], data["creditsUsed"]),
                    )

                    # Update User table (subtract creditsUsed from credits)
                    cursor.execute(
                        'UPDATE "User" SET "credits" = "credits" - %s WHERE "username" = %s',
                        (data["creditsUsed"], data["username"]),
                    )

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
