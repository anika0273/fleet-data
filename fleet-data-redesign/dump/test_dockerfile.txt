FROM python:3.11-slim

# Install Java, wget, unzip
RUN apt-get update && \
    apt-get install -y default-jdk wget unzip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir pyspark

# Copy project code
COPY etl ./etl
COPY config ./config
COPY analysis ./analysis
COPY lib ./lib
COPY data_generation ./data_generation

#No CMD here, as we will run specific scripts using docker-compose