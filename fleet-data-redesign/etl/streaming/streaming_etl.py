from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, round as spark_round
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, IntegerType, TimestampType
import os
import psycopg2

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fleet_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "1234")
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:9092")

def create_table_if_not_exists():
    """Create the fleet_stream_processed table if it doesn't exist."""
    create_table_query = """
    CREATE TABLE IF NOT EXISTS fleet_stream_processed (
        vehicle_id VARCHAR,
        driver_id VARCHAR,
        route_id VARCHAR,
        timestamp TIMESTAMP,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        hour_of_day INTEGER,
        weather VARCHAR,
        road_type VARCHAR,
        traffic_density VARCHAR,
        rush_hour BOOLEAN,
        vehicle_type VARCHAR,
        vehicle_age_years INTEGER,
        device_generation INTEGER,
        device_cost_usd DOUBLE PRECISION,
        driver_experience_years INTEGER,
        driver_training VARCHAR,
        optimized_route_flag BOOLEAN,
        speed DOUBLE PRECISION,
        distance_traveled_km DOUBLE PRECISION,
        idle_time_minutes DOUBLE PRECISION,
        fuel_consumption_liters DOUBLE PRECISION,
        fuel_rate_l_per_100km DOUBLE PRECISION,
        sensor_battery DOUBLE PRECISION,
        sensor_signal_strength DOUBLE PRECISION,
        data_latency_ms DOUBLE PRECISION,
        gps_accuracy_meters DOUBLE PRECISION,
        packet_loss_rate DOUBLE PRECISION,
        braking_event BOOLEAN,
        collision_alert BOOLEAN,
        lane_change_event BOOLEAN,
        harsh_acceleration_event BOOLEAN,
        sensor_fault_event BOOLEAN,
        network_delay_event BOOLEAN,
        gps_loss_event BOOLEAN,
        event_type VARCHAR,
        maintenance_type VARCHAR,
        time_since_last_maintenance_days DOUBLE PRECISION,
        breakdown_event BOOLEAN,
        downtime_hours DOUBLE PRECISION,
        maintenance_cost_usd DOUBLE PRECISION,
        intervention_active BOOLEAN,
        risk_label BOOLEAN,
        fuel_efficiency_l_per_km DOUBLE PRECISION
    );
    """
    conn = None
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        print("Table 'fleet_stream_processed' checked/created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_spark_session():
    postgres_jar_path = os.getenv("POSTGRES_JAR", "/app/lib/postgresql-42.7.3.jar")
    spark = (
        SparkSession.builder
        .appName("FleetStreamingETL")
        .config("spark.jars", postgres_jar_path)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

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
        StructField("risk_label", BooleanType()),
        StructField("fuel_efficiency_l_per_km", DoubleType())
    ])

def main():
    # Ensure the target postgres table exists before streaming starts
    create_table_if_not_exists()

    spark = get_spark_session()
    schema = define_schema()
    
    kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKERS) \
        .option("subscribe", "fleet-data") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    string_df = kafka_stream.selectExpr("CAST(value AS STRING)")

    fleet_df = string_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

    fleet_df = fleet_df.withColumn(
        "fuel_efficiency_l_per_km",
        spark_round(col("fuel_consumption_liters") / col("distance_traveled_km"), 3)
    )

    filtered_df = fleet_df.filter(col("speed") > 0)

    pg_url = f"jdbc:postgresql://postgres:5432/fleet_db"
    pg_properties = {
        "user": "postgres",
        "password": "1234",
        "driver": "org.postgresql.Driver"
    }

    def write_batch(df, epoch_id):
        count = df.count()
        print(f"[streaming_etl] epoch={epoch_id} rows={count}")
        df.select("vehicle_id","timestamp","speed","event_type").limit(3).show(truncate=False)
        if count > 0:
            try:
                df.write.jdbc(pg_url, "fleet_stream_processed", mode="append", properties=pg_properties)
                print(f"[streaming_etl] wrote {count} rows to Postgres")
            except Exception as e:
                print(f"[streaming_etl] ERROR writing to Postgres: {e}")

    query = (
        filtered_df.writeStream
        .foreachBatch(write_batch)
        .outputMode("append")
        .option("checkpointLocation", "/tmp/checkpoints/fleet_streaming_etl")  # important for stability
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("[streaming_etl] started Structured Streaming query; waiting for data…")
    query.awaitTermination()

if __name__ == "__main__":
    main()


# Start by removing old containers and rhe postgres volume: docker compose down -v \
# docker system prune -f
# Build the images: docker compose build --no-cache
# Start the services: docker compose up -d postgres kafka zookeeper
# docker ps
# docker compose up -d kafka-init OR do it manually: docker exec -it fleet-data-kafka \
#  kafka-topics --create --topic fleet-data --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
# List topics to confirm: docker exec -it fleet-data-kafka kafka-topics --list --bootstrap-server kafka:9092
#Start the producer (inside Docker): docker compose up -d producer   docker logs -f fleet-data-producer # you should see  Sent: {...} lines
# Start the streaming ETL job: docker compose up -d streaming-etl  docker logs -f fleet-data-streaming-etl # expect to see batches like: [streaming_etl] epoch=1 rows=30... wrote 30 rows to Postgres

# Check the target table in Postgres: docker exec -it fleet-data-postgres psql -U postgres -d fleet_db
# \dt
# select * from fleet_stream_processed limit 10;


# To monitor start the grafana and prometheus container: docker compose up -d grafana prometheus
# Access Grafana at http://localhost:3000 (admin/admin)
# Add Prometheus data source: http://localhost:9090/query

# in Grafana UI Create a new datasource:
# Name: Prometheus
# URL: http://prometheus:9090
# Save & Test

#
# In prometheus UI: http://localhost:9090/targets you should see the streaming_etl job as UP

