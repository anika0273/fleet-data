# streaming/producer.py
# This script generates synthetic fleet data and sends it to a Kafka topic.
from kafka import KafkaProducer   # Kafka client library for Python
import json
import time

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generation.generate_data import generate_fleet_data  # Your existing synthetic data generator

# -----------------------------
# Function to convert Python objects into JSON bytes for Kafka
# -----------------------------
def json_serializer(data):
    """
    Converts Python dict to JSON bytes for Kafka.
    Handles Pandas Timestamp objects by converting them to ISO format strings.
    """
    def default(o):
        if hasattr(o, 'isoformat'):
            return o.isoformat()  # Convert Timestamp to string
        raise TypeError(f"Type {type(o)} not serializable")

    return json.dumps(data, default=default).encode('utf-8')


# -----------------------------
# Kafka Producer function
# -----------------------------
def produce_stream(num_records=1000, batch_size=10, delay=1):
    """
    num_records: total messages to send (not strictly used here since it's an infinite loop)
    batch_size: how many records to send per batch
    delay: seconds to wait between batches
    """
    # Create Kafka producer instance
    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],  # Kafka broker address
        value_serializer=json_serializer       # Serialize messages as JSON
    )

    while True:
        # Generate synthetic fleet data as a Pandas DataFrame
        df_batch = generate_fleet_data(num_records=batch_size, fleet_size=50, driver_pool=40)

        # Iterate through each row and send to Kafka
        for _, record in df_batch.iterrows():
            data = record.to_dict()            # Convert row to Python dictionary
            producer.send('fleet-data', data)  # Send data to Kafka topic 'fleet-data'
            print(f"Sent: {data}")              # Log to console for debugging

        # Ensure all messages are actually sent to Kafka before next batch
        producer.flush()

        # Pause before sending the next batch
        time.sleep(delay)

# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    produce_stream()
# This script continuously generates synthetic fleet data and sends it to a Kafka topic.
# You can stop it with Ctrl+C when you want to end the stream.

# -----------------------------
# Make sure Kafka is running and the topic 'fleet-data' exists before running this script.
# command: docker exec -it $(docker ps -qf "name=kafka") kafka-topics \  --list \  --bootstrap-server localhost:9092

# To create the topic if it doesn't exist:
# docker exec -it $(docker ps -qf "name=kafka") kafka-topics \  --create \  --topic fleet-data \  --bootstrap-server localhost:9092 \  --partitions 1 \  --replication-factor 1

# You can adjust num_messages, batch_size, and delay parameters in produce_stream() as needed.
# -----------------------------
