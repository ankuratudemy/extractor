# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app
# Download the Tika app JAR from Apache Hub
RUN wget -O ./tika-app.jar --no-check-certificate https://downloads.apache.org/tika/3.0.0-BETA/tika-app-3.0.0-BETA.jar
# Install Java, wget, and SSL dependencies
RUN apt-get update && apt-get install -y --no-install-recommends default-jre wget ca-certificates openssl apt-transport-https software-properties-common
# Download the latest CA certificates
RUN update-ca-certificates

# Install the required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN echo "deb http://deb.debian.org/debian buster contrib" >> /etc/apt/sources.list
RUN echo "deb-src http://deb.debian.org/debian buster contrib" >> /etc/apt/sources.list

# RUN echo "deb http://deb.debian.org/debian/ bookworm contrib" >> /etc/apt/sources.list
# RUN echo "deb-src http://deb.debian.org/debian/ bookworm contrib" >> /etc/apt/sources.list

# RUN echo 'deb http://http.us.debian.org/debian jessie main contrib non-free' >> /etc/apt/sources.list
RUN cat /etc/apt/sources.list
# RUN apt-get update && apt-get install -y --no-install-recommends ttf-mscorefonts-installer
RUN apt-get update && apt-get install -y --no-install-recommends -t bookworm ttf-mscorefonts-installer 
RUN apt-get update && apt-get install -y libreoffice unoconv fontconfig
# RUN apt-get install -y fonts-roboto fonts-firacode fonts-open-sans
RUN rm -Rf /usr/share/fonts/*
RUN ls -ltr /usr/share/fonts
COPY /fonts/* /usr/share/fonts
RUN ls -ltr /usr/share/fonts
RUN fc-cache -f -v
RUN fc-list

# RUN wget -O find_uno.py --no-check-certificate https://gist.githubusercontent.com/regebro/036da022dc7d5241a0ee97efdf1458eb/raw/find_uno.py
# RUN python3 find_uno.py
COPY ./shared /app/shared

# Copy the application code to the container
COPY app.py .

# Create a new directory called "uploads"
RUN mkdir /app/uploads

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the entrypoint command to run the Flask application
CMD ["python3", "app.py"]