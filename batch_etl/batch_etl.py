"""
batch_etl.py

Batch ETL pipeline for processing synthetic fleet vehicle telemetry data.

ETL Flow:
---------
1. EXTRACT: Read raw fleet data from PostgreSQL via JDBC, applying pushdown filters 
            to reduce I/O and only fetch reasonable values.
2. TRANSFORM: Clean/normalize fields, remove bad data, 
              add derived fields (hour_of_day), enforce value constraints.
3. LOAD: Write cleaned data back into PostgreSQL 
         and into Parquet format for analytical use cases.

Why Spark?
----------
- Handles large datasets efficiently in-memory.
- Parallelizes JDBC reads/writes.
- Allows complex transformations and aggregations at scale.
"""

from pyspark.sql.functions import col, hour, lower, count, avg, when, round as spark_round
from utils import get_spark_session, get_jdbc_properties
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import traceback

# -----------------------
# Prometheus Metrics
# -----------------------
etl_job_duration = Histogram("fleet_batch_etl_duration_seconds", "Duration of ETL job in seconds")
rows_read = Gauge("fleet_batch_rows_read", "Number of rows read from Postgres")
rows_cleaned = Gauge("fleet_batch_rows_cleaned", "Number of rows after cleaning")
etl_failures = Counter("fleet_batch_etl_failures_total", "Number of failed ETL runs")


# -----------------------
# EXTRACT
# -----------------------
def read_from_postgres(spark, table_name):
    """
    Extract fleet data from Postgres with pushdown filtering.
    Filters applied:
    - GPS coordinates not null & within NYC bounds: ensures only valid positions.
    - Speed <= 160 km/h: realistic vehicle speeds.
    - Sensor battery 0–100%: physically possible battery values.
    - Fuel consumption and downtime positive: removes invalid data.
    """
    jdbc_conf = get_jdbc_properties()
    query = f"""
        (SELECT * 
         FROM {table_name} 
         WHERE latitude IS NOT NULL AND latitude BETWEEN 40.4774 AND 40.9176
           AND longitude IS NOT NULL AND longitude BETWEEN -74.2591 AND -73.7004
           AND speed IS NOT NULL AND speed >= 0 AND speed <= 160
           AND sensor_battery IS NOT NULL AND sensor_battery BETWEEN 0 AND 100
           AND fuel_consumption_liters IS NULL OR fuel_consumption_liters > 0
           AND downtime_hours IS NULL OR downtime_hours >= 0) AS subquery
    """
    return spark.read.jdbc(url=jdbc_conf["url"], table=query, properties=jdbc_conf["properties"])



# -----------------------
# TRANSFORM + FEATURE ENGINEERING
# -----------------------
def clean_and_transform(df):
    """
    Cleans and enriches dataset for ML and analytics.
    Transformations & Business Impact:
    - Cast numeric fields: ensures mathematical operations work.
    - Normalize text fields (weather, road_type, traffic_density, driver_training): avoids split categories.
    - Derived features:
        - hour_of_day: detects time-of-day patterns affecting risk & events (driver intervention, route optimization).
        - vehicle_age_group: categorizes vehicle age to model predictive maintenance & fuel consumption.
        - fuel_efficiency_l_per_km: supports route optimization and fuel analysis.
        - downtime_flag: binary target for predictive maintenance analysis.
    - Filters negative/unrealistic fuel or downtime values.
    """

    df_clean = df.withColumn("sensor_battery", col("sensor_battery").cast("double"))
    df_clean = df_clean.withColumn("speed", col("speed").cast("double"))
    df_clean = df_clean.withColumn("fuel_consumption_liters", col("fuel_consumption_liters").cast("double"))
    df_clean = df_clean.withColumn("downtime_hours", col("downtime_hours").cast("double"))

    # Remove physically impossible values
    df_clean = df_clean.filter((col("sensor_battery") >= 0) & (col("sensor_battery") <= 100))
    df_clean = df_clean.filter((col("speed") >= 0) & (col("speed") <= 160))
    df_clean = df_clean.filter((col("fuel_consumption_liters") >= 0) | col("fuel_consumption_liters").isNull())
    df_clean = df_clean.filter((col("downtime_hours") >= 0) | col("downtime_hours").isNull())

    # Normalize categorical fields
    for c in ["weather", "road_type", "traffic_density", "driver_training"]:
        df_clean = df_clean.withColumn(c, lower(col(c)))

    # Derived feature: hour_of_day
    if "hour_of_day" not in df_clean.columns:
        df_clean = df_clean.withColumn("hour_of_day", hour(col("timestamp")))

    # Derived feature: vehicle_age_group (0-3, 4-6, 7-9, 10+)
    df_clean = df_clean.withColumn("vehicle_age_group", 
                                   when(col("vehicle_age_years") <= 3, "0-3")
                                   .when((col("vehicle_age_years") >=4) & (col("vehicle_age_years") <=6), "4-6")
                                   .when((col("vehicle_age_years") >=7) & (col("vehicle_age_years") <=9), "7-9")
                                   .otherwise("10+"))

    # Derived feature: fuel efficiency (L/km)
    df_clean = df_clean.withColumn(
    "fuel_efficiency_l_per_km",
    spark_round(col("fuel_consumption_liters") / col("distance_traveled_km"), 2)
)

    # Derived feature: downtime_flag (binary)
    df_clean = df_clean.withColumn("downtime_flag", when(col("downtime_hours") > 0, 1).otherwise(0))

    return df_clean


def aggregate_metrics(df):
    """
    Aggregates metrics useful for analytics dashboards & data quality checks.
    Metrics:
    - Event counts and averages (speed, battery, fuel efficiency)
    - Supports driver interventions, route optimization, and predictive maintenance insights
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
def write_to_postgres(df, table_name):
    """
    Load Step: Writes a DataFrame to PostgreSQL in overwrite mode.

    Uses a batch size to improve performance.
    This creates/overwrites a cleaned version of the data for DB querying.
    """
    jdbc_conf = get_jdbc_properties()
    props = {**jdbc_conf["properties"], "batchsize": "10000"}
    df.write.jdbc(url=jdbc_conf["url"], table=table_name, mode="overwrite", properties=props)


def write_to_parquet(df, destination_path):
    """
    Load Step: Writes the cleaned dataset to a Parquet file.

    Parquet benefits:
    - Columnar storage for efficient analytics.
    - Can be read by Spark, Pandas, and BI tools.
    - Efficient for distributed querying.
    """
    df.write.mode("overwrite").parquet(destination_path)


# -----------------------
# MAIN ETL PIPELINE
# -----------------------
@etl_job_duration.time()
def main():
    try:
        spark = get_spark_session()
        spark.sparkContext.setLogLevel("WARN")

        # 1. EXTRACT
        print("📥 Reading filtered fleet data from Postgres...")
        df_raw = read_from_postgres(spark, "fleet_data")
        rows_read.set(df_raw.count())

        # 2. TRANSFORM
        print("🧹 Cleaning & transforming data...")
        df_clean = clean_and_transform(df_raw).cache()
        rows_cleaned.set(df_clean.count())

        # 3. AGGREGATION
        print("📊 Aggregating metrics...")
        metrics_df = aggregate_metrics(df_clean)
        metrics_df.show()

        # 4. LOAD
        print("💾 Writing cleaned data to Postgres & Parquet...")
        write_to_postgres(df_clean, "fleet_data_cleaned")
        write_to_parquet(df_clean, "data/output/cleaned_fleet_data.parquet")

        spark.stop()
        print("✅ ETL job completed successfully.")

    except Exception as e:
        etl_failures.inc()
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    # Start Prometheus HTTP server for metrics
    start_http_server(8000)  # Prometheus will scrape http://localhost:8000/metrics
    main()


"""
What This Does:

    Runs your ETL job as before.

    Starts a small web server at http://localhost:8000/metrics.

    Exposes metrics like (example values):
        fleet_batch_rows_read  50000
        fleet_batch_rows_cleaned  48000
        fleet_batch_etl_duration_seconds  42.7
        fleet_batch_etl_failures_total  0

"""