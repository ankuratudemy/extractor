# Use the official Python image as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install Java
RUN apt-get update && apt-get install -y default-jre

# Copy the requirements file to the container

COPY requirements.txt .
COPY tika-app.jar .
COPY ./shared /app/shared
# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY app.py .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint command to run the Flask application
CMD ["python", "app.py"]