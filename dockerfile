# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install Java, wget, and SSL dependencies
RUN apt-get update && apt-get install -y default-jre wget ca-certificates openssl

# Download the latest CA certificates
RUN update-ca-certificates

# Download the Tika app JAR from Apache Hub
RUN wget -O ./tika-app.jar --no-check-certificate https://downloads.apache.org/tika/tika-app-1.29.jar


COPY requirements.txt .
COPY ./shared /app/shared
# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY app.py .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint command to run the Flask application
CMD ["python", "app.py"]