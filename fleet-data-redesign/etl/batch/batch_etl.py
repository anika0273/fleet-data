"""
etl/batch/batch_etl.py

Batch ETL pipeline for processing synthetic fleet vehicle telemetry data.

ETL Flow:
---------
1. EXTRACT:
   - Read raw fleet data from PostgreSQL via JDBC
   - Push filters down to DB to reduce I/O
   - Ensures we only fetch valid, reasonable records

2. TRANSFORM:
   - Clean and normalize fields
   - Add derived fields (hour_of_day, vehicle_age_group, efficiency, downtime_flag)
   - Remove unrealistic/bad values

3. LOAD:
   - Write cleaned data back to PostgreSQL (overwrite table)
   - Also write to Parquet for downstream analysis/ML use cases

Observability:
--------------
- Exposes Prometheus metrics on http://localhost:8001/metrics
- Metrics include ETL duration, row counts, and failure count
"""

import traceback
import time
from pyspark.sql.functions import (
    col, hour, lower, count, avg, when, round as spark_round
)
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Import helpers from the same package
from etl.batch.utils import get_spark_session, get_jdbc_properties

# -----------------------
# Prometheus Metrics
# -----------------------
etl_job_duration = Histogram(
    "fleet_batch_etl_duration_seconds",
    "Duration of ETL job in seconds"
)
rows_read = Gauge(
    "fleet_batch_rows_read",
    "Number of rows read from Postgres"
)
rows_cleaned = Gauge(
    "fleet_batch_rows_cleaned",
    "Number of rows after cleaning"
)
etl_failures = Counter(
    "fleet_batch_etl_failures_total",
    "Number of failed ETL runs"
)


# -----------------------
# EXTRACT
# -----------------------
def read_from_postgres(spark, table_name: str):
    """
    Extract fleet data from Postgres with filters applied.

    Pushdown filters (run inside DB, not Spark):
    - GPS coordinates within NYC bounds
    - Speed ≤ 160 km/h
    - Battery within 0–100%
    - Fuel consumption > 0 or NULL
    - Downtime ≥ 0 or NULL
    """
    jdbc_conf = get_jdbc_properties()
    query = f"""
        (SELECT * 
         FROM {table_name} 
         WHERE latitude IS NOT NULL AND latitude BETWEEN 40.4774 AND 40.9176
           AND longitude IS NOT NULL AND longitude BETWEEN -74.2591 AND -73.7004
           AND speed IS NOT NULL AND speed >= 0 AND speed <= 160
           AND sensor_battery IS NOT NULL AND sensor_battery BETWEEN 0 AND 100
           AND (fuel_consumption_liters IS NULL OR fuel_consumption_liters > 0)
           AND (downtime_hours IS NULL OR downtime_hours >= 0)) AS subquery
    """
    return spark.read.jdbc(
        url=jdbc_conf["url"],
        table=query,
        properties=jdbc_conf["properties"]
    )


# -----------------------
# TRANSFORM + FEATURE ENGINEERING
# -----------------------
def clean_and_transform(df):
    """
    Cleans and enriches dataset for ML and analytics.

    Key transforms:
    - Cast numeric fields
    - Normalize categorical text
    - Add derived features:
        * hour_of_day
        * vehicle_age_group
        * fuel_efficiency_l_per_km
        * downtime_flag
    """
    # Cast numeric fields explicitly
    df_clean = df.withColumn("sensor_battery", col("sensor_battery").cast("double"))
    df_clean = df_clean.withColumn("speed", col("speed").cast("double"))
    df_clean = df_clean.withColumn("fuel_consumption_liters", col("fuel_consumption_liters").cast("double"))
    df_clean = df_clean.withColumn("downtime_hours", col("downtime_hours").cast("double"))

    # Remove impossible values
    df_clean = df_clean.filter((col("sensor_battery") >= 0) & (col("sensor_battery") <= 100))
    df_clean = df_clean.filter((col("speed") >= 0) & (col("speed") <= 160))
    df_clean = df_clean.filter((col("fuel_consumption_liters") >= 0) | col("fuel_consumption_liters").isNull())
    df_clean = df_clean.filter((col("downtime_hours") >= 0) | col("downtime_hours").isNull())

    # Normalize categorical fields
    for c in ["weather", "road_type", "traffic_density", "driver_training"]:
        if c in df_clean.columns:
            df_clean = df_clean.withColumn(c, lower(col(c)))

    # Derived feature: hour_of_day
    if "timestamp" in df_clean.columns:
        df_clean = df_clean.withColumn("hour_of_day", hour(col("timestamp")))

    # Derived feature: vehicle_age_group
    df_clean = df_clean.withColumn(
        "vehicle_age_group", 
        when(col("vehicle_age_years") <= 3, "0-3")
        .when((col("vehicle_age_years") >= 4) & (col("vehicle_age_years") <= 6), "4-6")
        .when((col("vehicle_age_years") >= 7) & (col("vehicle_age_years") <= 9), "7-9")
        .otherwise("10+")
    )

    # Derived feature: fuel efficiency (L/km)
    if "distance_traveled_km" in df_clean.columns:
        df_clean = df_clean.withColumn(
            "fuel_efficiency_l_per_km",
            spark_round(col("fuel_consumption_liters") / col("distance_traveled_km"), 2)
        )

    # Derived feature: downtime_flag
    df_clean = df_clean.withColumn("downtime_flag", when(col("downtime_hours") > 0, 1).otherwise(0))

    return df_clean


def aggregate_metrics(df):
    """
    Aggregates metrics useful for analytics dashboards.

    Metrics include:
    - Event counts
    - Average speed, battery, fuel efficiency
    """
    return df.groupBy("event_type").agg(
        count("*").alias("event_count"),
        avg("speed").alias("avg_speed"),
        avg("sensor_battery").alias("avg_battery"),
        avg("fuel_efficiency_l_per_km").alias("avg_fuel_efficiency")
    )


# -----------------------
# LOAD
# -----------------------
def write_to_postgres(df, table_name: str):
    """
    Write cleaned DataFrame to PostgreSQL (overwrite mode).
    """
    jdbc_conf = get_jdbc_properties()
    props = {**jdbc_conf["properties"], "batchsize": "10000"}
    df.write.jdbc(
        url=jdbc_conf["url"],
        table=table_name,
        mode="overwrite",
        properties=props
    )


def write_to_parquet(df, destination_path: str):
    """
    Write cleaned DataFrame to Parquet for downstream analysis.
    """
    df.write.mode("overwrite").parquet(destination_path)


# -----------------------
# MAIN ETL PIPELINE
# -----------------------
@etl_job_duration.time()
def main():
    """
    Full ETL execution:
    - Extract → Transform → Aggregate → Load
    """
    try:
        spark = get_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        # 1. Extract
        print("📥 Reading filtered fleet data from Postgres...")
        df_raw = read_from_postgres(spark, "fleet_data")
        rows_read.set(df_raw.count())

        # 2. Transform
        print("🧹 Cleaning & transforming data...")
        df_clean = clean_and_transform(df_raw).cache()
        rows_cleaned.set(df_clean.count())

        # 3. Aggregate
        print("📊 Aggregating metrics...")
        metrics_df = aggregate_metrics(df_clean)
        metrics_df.show()

        # 4. Load
        print("💾 Writing cleaned data to Postgres & Parquet...")
        write_to_postgres(df_clean, "fleet_data_cleaned")
        write_to_parquet(df_clean, "data/output/cleaned_fleet_data.parquet")

        spark.stop()
        print("✅ ETL job completed successfully.")

    except Exception:
        etl_failures.inc()
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Start Prometheus HTTP server for metrics
    # Exposed on port 8001 (matches prometheus.yml scrape config)
    start_http_server(8001)
    main()


# docker compose up -d batch_etl 
# docker compose run --rm batch_etl python -m etl.batch.batch_etl