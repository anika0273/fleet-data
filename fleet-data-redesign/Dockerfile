FROM apache/airflow:2.9.0

USER root

# Install Java, wget, unzip, libgomp1 needed for PySpark
RUN apt-get update && \
    apt-get install -y default-jdk wget unzip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment for PySpark
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$JAVA_HOME/bin:$PATH"

USER airflow

# Copy requirements and install Python dependencies (including pyspark)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pyspark

# Copy project source code directories into container
COPY etl ./etl
COPY config ./config
COPY analysis ./analysis
COPY lib ./lib
COPY data_generation ./data_generation

# Set working directory to Airflow home (standard in official image)
WORKDIR /opt/airflow

# No CMD or ENTRYPOINT needed; handled by docker-compose commands
