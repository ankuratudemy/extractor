import os
import sys
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
    Example method from your original code. Left intact.
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
                    (subscription_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    connection.commit()
                    return 0
                else:
                    remaining_credits = row[0]
                    connection.commit()
                    return remaining_credits

    except Exception as e:
        print(f"Error connecting to PostgreSQL or retrieving credits: {str(e)}")
        sys.exit("Function execution failed.")


def update_file_status(file_id, status, page_count, updated_at):
    """
    Updates the status, pageCount, and updatedAt fields for a file in the File table.
    Based on your original example, but matching your new schema:
      - file_id -> "id" (String)
      - status -> "status" (String)
      - page_count -> "pageCount" (Int)
      - updated_at -> "updatedAt" (DateTime)
    """
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                try:
                    print(
                        f"file_id: {file_id}, status: {status}, "
                        f"page_count: {page_count}, updated_at: {updated_at}"
                    )
                    if not file_id or not status:
                        raise ValueError(
                            "'file_id' and 'status' are required."
                        )

                    cursor.execute(
                        """
                        UPDATE "File"
                        SET "status" = %s, "pageCount" = %s, "updatedAt" = %s
                        WHERE "id" = %s
                        """,
                        (status, page_count, updated_at, file_id),
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


def add_new_file(
    file_id,
    status,
    created_at,
    updated_at,
    name,
    file_type,
    file_key,
    page_count,
    upload_url,
    source_type,
    data_source_id,
    project_id,
):
    """
    Inserts a new record into the File table using the schema:
      - id          String  (PK, default uuid)
      - status      String
      - createdAt   DateTime
      - updatedAt   DateTime
      - name        String
      - type        String
      - key         String  (unique)
      - pageCount   Int
      - uploadUrl   String
      - sourceType  String
      - dataSourceId String
      - projectId   String
    """
    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                try:
                    print(
                        f"Adding new file: "
                        f"id={file_id}, status={status}, createdAt={created_at}, updatedAt={updated_at}, "
                        f"name={name}, type={file_type}, key={file_key}, pageCount={page_count}, "
                        f"uploadUrl={upload_url}, sourceType={source_type}, projectId={project_id}, dataSourceId={data_source_id}"
                    )

                    cursor.execute(
                        """
                        INSERT INTO "File" (
                            "id", "status", "createdAt", "updatedAt",
                            "name", "type", "key", "pageCount",
                            "uploadUrl", "sourceType", "projectId", "dataSourceId"
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            file_id,
                            status,
                            created_at,
                            updated_at,
                            name,
                            file_type,
                            file_key,
                            page_count,
                            upload_url,
                            source_type,
                            project_id,
                            data_source_id
                        ),
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


def update_file_by_id(file_id, **kwargs):
    """
    Dynamically updates one or more columns in the File table for the given file_id.
    Provide any column_name=value pairs via kwargs.

    Example:
        update_file_by_id(
            "some-uuid",
            status='finished',
            pageCount=5,
            name='UpdatedName'
        )
    """
    if not file_id:
        raise ValueError("File ID is required to update the record.")
    if not kwargs:
        print("No columns to update.")
        return

    columns = []
    values = []
    for col, val in kwargs.items():
        columns.append(f'"{col}" = %s')
        values.append(val)

    set_clause = ", ".join(columns)
    values.append(file_id)  # For the WHERE clause

    query = f'UPDATE "File" SET {set_clause} WHERE "id" = %s'

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                try:
                    print(f"Updating file {file_id} with fields: {kwargs}")
                    cursor.execute(query, values)
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


def delete_file_by_id(file_id):
    """
    Deletes a record from the File table by id.
    """
    if not file_id:
        raise ValueError("File ID is required to delete the record.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                try:
                    print(f"Deleting file with id={file_id}")
                    cursor.execute(
                        'DELETE FROM "File" WHERE "id" = %s',
                        (file_id,),
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


def delete_files_by_ids(file_ids):
    """
    Deletes multiple records from the File table by a list of ids.
    """
    if not file_ids:
        print("No file IDs provided for deletion.")
        return

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                connection.autocommit = False
                lock_id = int(time.time() * 1000)
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

                try:
                    print(f"Deleting files with ids={file_ids}")
                    cursor.execute(
                        'DELETE FROM "File" WHERE "id" = ANY(%s)',
                        (file_ids,),
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


def fetch_file_by_id(file_id):
    """
    Retrieves a single file record by its id. Returns None if not found.
    """
    if not file_id:
        raise ValueError("file_id is required.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM "File" WHERE "id" = %s',
                    (file_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                col_names = [desc[0] for desc in cursor.description]
                return dict(zip(col_names, row))

    except Exception as e:
        print(f"Error connecting to PostgreSQL or fetching file by id: {str(e)}")
        sys.exit("Function execution failed.")


def fetch_files_by_project_id_and_source_type(project_id, source_type):
    """
    Retrieves all files belonging to a specific projectId.
    """
    if not project_id:
        raise ValueError("project_id is required.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM "File" WHERE "projectId" = %s AND "sourceType" = %s',
                    (project_id, source_type),
                )
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

                results = []
                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    results.append(row_dict)

                return results

    except Exception as e:
        print(f"Error connecting to PostgreSQL or fetching files by projectId: {str(e)}")
        sys.exit("Function execution failed.")


def fetch_files_by_data_source_id(data_source_id):
    """
    Retrieves all files belonging to a specific data_source_id.
    """
    if not data_source_id:
        raise ValueError("data_source_id is required.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM "File" WHERE "dataSourceId" = %s',
                    (data_source_id,),
                )
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description]

                results = []
                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    results.append(row_dict)

                return results

    except Exception as e:
        print(f"Error connecting to PostgreSQL or fetching files by projectId: {str(e)}")
        sys.exit("Function execution failed.")


def get_project_details(project_id):
    """
    Retrieves projectId details.
    """
    if not project_id:
        raise ValueError("project_id is required.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM "Project" WHERE "id" = %s',
                    (project_id,),
                )
                 # Fetch one row from the result
                row = cursor.fetchone()
                # Retrieve column names from cursor description
                col_names = [desc[0] for desc in cursor.description]
                
                # Create a dictionary mapping column names to row values
                project_details = dict(zip(col_names, row))
                return project_details

    except Exception as e:
        print(f"Error connecting to PostgreSQL or fetching files by projectId: {str(e)}")
        sys.exit("Function execution failed.")

def get_data_source_details(data_source_id):
    """
    Retrieves data_source_id details.
    """
    if not data_source_id:
        raise ValueError("data_source_id is required.")

    try:
        with psycopg2.connect(**db_params) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT * FROM "DataSource" WHERE "id" = %s',
                    (data_source_id,),
                )
                 # Fetch one row from the result
                row = cursor.fetchone()
                # Retrieve column names from cursor description
                col_names = [desc[0] for desc in cursor.description]
                
                # Create a dictionary mapping column names to row values
                data_source_details = dict(zip(col_names, row))
                return data_source_details

    except Exception as e:
        print(f"Error connecting to PostgreSQL or fetching files by data_source_id: {str(e)}")
        sys.exit("Function execution failed.")
