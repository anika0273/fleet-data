# Section 4: Real-Time Data Streaming & Processing

(Kafka + Spark Structured Streaming)

## What Is This Section?

This part of the project demonstrates how to stream "live" vehicle telemetry data using Apache Kafka (as a message broker) and process it in near real-time using Apache Spark Structured Streaming.
It’s how big companies (like Uber, Tesla, Lyft) monitor fleets and react instantly to events, risks, or breakdowns!

## Overview: How It Works

- Kafka & Zookeeper — Manage real-time data streams.
- Kafka Producer (producer.py) — Continuously sends simulated vehicle data messages into Kafka.
- Spark Structured Streaming (streaming_etl.py) — Listens to Kafka, processes and filters the live data, then writes results to PostgreSQL and/or cloud storage.

Everything is modular—easy to change, test, and expand!

## Prerequisites

- Docker Desktop (for launching Kafka and Zookeeper containers)
- Python 3.8+
- Java 11+ (for Spark)
- Apache Spark and PySpark
- PostgreSQL running (for storing results)
- PostgreSQL JDBC driver in your lib/ folder
- Optional: AWS CLI and credentials (for writing to S3)

## Step-by-Step Instructions

1.  Start Kafka & Zookeeper Services

Navigate to your project’s root folder, then run (Terminal 1):

```bash
docker-compose -f docker-compose.kafka.yml up -d
```

Check if the services are running:

```bash
docker-compose -f docker-compose.kafka.yml ps
```

2.  Create a Kafka Topic for Fleet Data

Still in Terminal 1, run:

```bash
docker exec -it $(docker ps -qf "name=kafka") \
  kafka-topics --create --topic fleet-data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

This sets up a "mailbox" (topic) for vehicle data messages.

Ypu can verify by running this:

```bash
docker exec -it $(docker ps -qf "name=kafka") kafka-topics \
  --list \
  --bootstrap-server localhost:9092

```

3.  Start the Kafka Producer

Open a new terminal window (Terminal 2), activate your Python environment, navigate to project root, then run:

```python
python streaming/producer.py
```

This script generates synthetic vehicle records and streams them, one-by-one, into the fleet-data Kafka topic.

Example output:

```text
Sent: {'vehicle_id': 'ABC123', 'speed': 56, ...}
```

4.  Start the Spark Structured Streaming Consumer

In a third terminal window (Terminal 3), start your streaming Spark job (check if the versions are compatible here: https://mvnrepository.com/artifact/org.apache.spark/spark-sql-kafka-0-10_2.13):

```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0-preview2 \
  --jars lib/postgresql-42.7.3.jar \
  streaming/streaming_etl.py
```

But before

1. Check your Spark version:

```bash
spark-submit --version
```

Green Signal: Output shows Spark version 4.0.0 or compatible with your setup

2. Check Scala version used by Spark

```bash
spark-shell
```

Green signal: Scala version matches 2.13.x because Spark 4.0.0 is built for Scala 2.13. If you get 2.12 or other, choose compatible Kafka package for that version.

What happens here:

- Subscribes to the fleet-data Kafka topic.
- Processes live messages (filtering, calculating metrics, etc.).
- Writes each micro-batch to PostgreSQL (fleet_stream_processed table).
- Optionally, can be adapted to save data to S3 by updating the output path and AWS settings.

## What Each Script Does

- producer.py

  - Simulates and streams new vehicle records into Kafka every second.
  - Each message is a JSON object representing real telemetry data (speed, location, sensor status, etc).

- streaming_etl.py
  - Uses Spark Structured Streaming to read each new message from Kafka as it arrives and parse, filter, and process the data on-the-fly.
  - Write cleaned/filtered data into PostgreSQL (or S3/local storage if configured).

## How to Stop Everything

- In each terminal running a script, press Ctrl + C.
- To stop Kafka and Zookeeper containers:

```bash
docker-compose -f docker-compose.kafka.yml down
```
