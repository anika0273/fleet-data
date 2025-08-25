# etl/streaming/producer.py
# producer.py sends JSON events to Kafka.
from kafka import KafkaProducer
import json
import time
import sys
import os

# Allow imports from project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


print("Current directory:", os.getcwd())
print("Module search paths:", sys.path)
print("Folder content:", os.listdir(os.path.join(os.getcwd(), "data_generation")))

from data_generation.generate_data import generate_fleet_data

# -----------------------------
# Function to convert Python objects into JSON bytes for Kafka
# -----------------------------
def json_serializer(data):
    def default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(f"Type {type(o)} not serializable")
    return json.dumps(data, default=default).encode("utf-8")


# -----------------------------
# Kafka Producer function
# -----------------------------
def produce_stream(num_records=1000, batch_size=10, delay=1):
    brokers = os.getenv("KAFKA_BROKERS", "localhost:9092")  # host default
    print(f"[producer] bootstrap_servers = {brokers}")
    producer = KafkaProducer(
        bootstrap_servers=[brokers],
        value_serializer=json_serializer
    )
    while True:
        df_batch = generate_fleet_data(num_records=batch_size, fleet_size=50, driver_pool=40)
        for _, record in df_batch.iterrows():
            producer.send("fleet-data", record.to_dict())
            print(f"Sent: {record.to_dict()}")
        producer.flush()
        time.sleep(delay)

if __name__ == "__main__":
    produce_stream()
