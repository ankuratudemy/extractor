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


def get_remaining_credits(subscription_id):
    """
    Returns the remaining credits for a given subscription_id from the Subscription table.
    """
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)  # Advisory lock ID
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                cursor.execute(
                    'SELECT "remainingCredits" FROM "Subscription" WHERE "id" = %s',
                    (subscription_id,)
                )
                row = cursor.fetchone()
                if row is None:
                    # If no row found, treat as 0 or handle as error
                    connection.commit()
                    return 0
                else:
                    remaining_credits = row[0]
                    connection.commit()
                    return remaining_credits

    except Exception as e:
        print(
            f"Error connecting to PostgreSQL or retrieving credits: {str(e)}")
        sys.exit("Function execution failed.")


def update_file_status(id, status, page_nums, updatedAt):
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False

                try:
                    print(
                        f" id: {id} status: {status} page_nums {page_nums} updatedAt {updatedAt}")
                    if not id or not status:
                        raise ValueError(
                            "Either 'status', 'id', or 'page_nums' is missing.")

                    # milliseconds since epoch
                    lock_id = int(time.time() * 1000)
                    cursor.execute(
                        "SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                    cursor.execute(
                        'UPDATE "File" SET "status" = %s, "pageCount" = %s, "updatedAt" = %s WHERE "id" = %s',
                        (status, page_nums, updatedAt, id),
                    )
                    connection.commit()
                    print("Transaction completed successfully.")
                    return "success"

                except Exception as e:
                    connection.rollback()
                    print(f"Error during transaction: {str(e)}")
                    sys.exit("Function execution failed.")

    except Exception as e:
        print(f"Error connecting to PostgreSQL: {str(e)}")
        sys.exit("Function execution failed.")
