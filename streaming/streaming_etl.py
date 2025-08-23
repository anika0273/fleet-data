# streaming/streaming_etl.py
# Spark Structured Streaming consumes Kafka → applies schema → filters → writes to Postgres table fleet_stream_processed.
# This script sets up a Spark session, reads streaming data from Kafka, applies transformations, and writes the results to PostgreSQL.

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, IntegerType, TimestampType

def get_spark_session():
    return SparkSession.builder \
        .appName("FleetStreamingETL") \
        .config("spark.jars", "/Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar") \
        .getOrCreate()

def define_schema():
    return StructType([
        StructField("vehicle_id", StringType()),
        StructField("driver_id", StringType()),
        StructField("route_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("hour_of_day", IntegerType()),
        StructField("weather", StringType()),
        StructField("road_type", StringType()),
        StructField("traffic_density", StringType()),
        StructField("rush_hour", BooleanType()),
        StructField("vehicle_type", StringType()),
        StructField("vehicle_age_years", IntegerType()),
        StructField("device_generation", IntegerType()),
        StructField("device_cost_usd", DoubleType()),
        StructField("driver_experience_years", IntegerType()),
        StructField("driver_training", StringType()),
        StructField("optimized_route_flag", BooleanType()),
        StructField("speed", DoubleType()),
        StructField("distance_traveled_km", DoubleType()),
        StructField("idle_time_minutes", DoubleType()),
        StructField("fuel_consumption_liters", DoubleType()),
        StructField("fuel_rate_l_per_100km", DoubleType()),
        StructField("sensor_battery", DoubleType()),
        StructField("sensor_signal_strength", DoubleType()),
        StructField("data_latency_ms", DoubleType()),
        StructField("gps_accuracy_meters", DoubleType()),
        StructField("packet_loss_rate", DoubleType()),
        StructField("braking_event", BooleanType()),
        StructField("collision_alert", BooleanType()),
        StructField("lane_change_event", BooleanType()),
        StructField("harsh_acceleration_event", BooleanType()),
        StructField("sensor_fault_event", BooleanType()),
        StructField("network_delay_event", BooleanType()),
        StructField("gps_loss_event", BooleanType()),
        StructField("event_type", StringType()),
        StructField("maintenance_type", StringType()),
        StructField("time_since_last_maintenance_days", DoubleType()),
        StructField("breakdown_event", BooleanType()),
        StructField("downtime_hours", DoubleType()),
        StructField("maintenance_cost_usd", DoubleType()),
        StructField("intervention_active", BooleanType()),
        StructField("risk_label", BooleanType())
    ])

def main():
    spark = get_spark_session()
    schema = define_schema()

    # Read streaming data from Kafka
    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "fleet-data") \
        .option("startingOffsets", "earliest") \
        .load()

    # Convert the binary value column to string
    string_df = kafka_stream.selectExpr("CAST(value AS STRING)")

    # Parse JSON and apply schema
    fleet_df = string_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

    # Example transformation: Calculate fuel efficiency (L/km) for ML-ready analysis
    fleet_df = fleet_df.withColumn(
        "fuel_efficiency_l_per_km",
        spark_round(col("fuel_consumption_liters") / col("distance_traveled_km"), 3)
    )

    # Example filter: keep records with realistic speed > 0
    filtered_df = fleet_df.filter(col("speed") > 0)

    # PostgreSQL connection options
    pg_url = "jdbc:postgresql://postgres:5432/fleet_db"
    pg_properties = {
        "user": "postgres",
        "password": "1234",
        "driver": "org.postgresql.Driver"
    }

    # Write streaming data to PostgreSQL
    query = filtered_df.writeStream \
        .foreachBatch(lambda df, epochId: df.write.jdbc(pg_url, "fleet_stream_processed", mode="append", properties=pg_properties)) \
        .outputMode("update") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    main()


# Run and Test Instructions:
# 1. Ensure PostgreSQL is running and the fleet_db database exists.
# Start Zookeeper and Kafka using Docker: docker-compose -f docker-compose.kafka.yml up -d
# 2. Create the 'fleet-data' topic if it doesn't exist:
# docker exec -it $(docker ps -qf "name=kafka") kafka-topics --create --topic fleet-data --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
# 3. Start the Kafka producer to send data: python -m streaming/producer.py
# 4. In a separate terminal, start Spark Streaming job: Run this script: spark-submit \
#     --jars /Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0 \
#     streaming/streaming_etl.py 
# 5. Monitor the PostgreSQL database to see the processed data in the 'fleet_stream_processed' table.