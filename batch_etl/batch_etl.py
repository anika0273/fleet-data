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

from pyspark.sql.functions import col, hour, lower, count, avg
from utils import get_spark_session, get_jdbc_properties
import time

# -----------------------
# EXTRACT
# -----------------------
def read_from_postgres(spark, table_name):
    """
    Extract Step: Reads fleet data from PostgreSQL using JDBC.

    Pushdown filtering is used so that:
    - Rows with missing GPS (latitude/longitude) are removed at database level.
    - Rows with missing speed are removed at database level.
    - Only rows with realistic speeds (<= 200 km/h) are fetched.

    This reduces the volume of data transferred to Spark, improving efficiency.
    """
    jdbc_conf = get_jdbc_properties()

    # SQL subquery with initial filters BEFORE Spark reads the data
    query = f"""
        (SELECT * 
         FROM {table_name} 
         WHERE latitude IS NOT NULL 
           AND longitude IS NOT NULL 
           AND speed IS NOT NULL 
           AND speed <= 200) AS subquery
    """
    return spark.read.jdbc(url=jdbc_conf["url"], table=query, properties=jdbc_conf["properties"])


# -----------------------
# TRANSFORM
# -----------------------
def clean_and_transform(df):
    """
    Transform Step: Cleans and enriches the dataset.

    Transformations performed:
    - Casts `sensor_battery` to double to ensure numeric operations work.
    - Removes rows where battery is outside realistic bounds (0–100%).
    - Normalizes categorical `weather` values to lowercase for consistency.
    - Adds `hour_of_day` field from `timestamp` if not already present 
      (useful for time-of-day analysis).

    Why:
    - Type casting ensures downstream aggregates & comparisons work without errors.
    - Filtering out physically impossible battery/speed values reduces noise.
    - Normalizing categories prevents splitting values like 'Sunny' vs 'sunny'.
    - Adding derived features increases analytical flexibility.
    """
    # Ensure correct data type for battery level
    df_clean = df.withColumn("sensor_battery", col("sensor_battery").cast("double"))

    # Filter out physically impossible battery readings
    df_clean = df_clean.filter((col("sensor_battery") >= 0) & (col("sensor_battery") <= 100))

    # Normalize text fields for consistent grouping in aggregations
    df_clean = df_clean.withColumn("weather", lower(col("weather")))

    # Add derived `hour_of_day` for time-based analysis if not already present
    if "hour_of_day" not in df_clean.columns:
        df_clean = df_clean.withColumn("hour_of_day", hour(col("timestamp")))

    return df_clean


def aggregate_metrics(df):
    """
    Transform (Aggregation) Step:
    Generates high-level metrics from the cleaned dataset.

    Metrics:
    - Event counts: Number of records per `event_type`
    - Average speed per `event_type`
    - Average battery level per `event_type`

    Why:
    - Helps quickly assess frequency of different event types in the dataset.
    - Shows operational insights (e.g., which event types correlate with high/low speeds).
    - Supports data quality monitoring through average values.
    """
    return df.groupBy("event_type").agg(
        count("*").alias("event_count"),
        avg("speed").alias("avg_speed"),
        avg("sensor_battery").alias("avg_battery")
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
def main():
    # Initialize Spark session
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    start_time = time.time()

    # 1. EXTRACT
    print("📥 Reading filtered fleet data from Postgres...")
    df_raw = read_from_postgres(spark, "fleet_data")
    print(f"Loaded {df_raw.count()} rows after initial pushdown filtering.")

    # 2. TRANSFORM
    print("🧹 Cleaning and transforming data...")
    df_clean = clean_and_transform(df_raw).cache()  # Cache to reuse cleaned dataset in multiple stages
    print(f"Rows after cleaning: {df_clean.count()}")

    print("📊 Aggregating metrics...")
    metrics_df = aggregate_metrics(df_clean)
    metrics_df.show()

    # 3. LOAD
    print("💾 Writing cleaned data back to PostgreSQL and Parquet...")
    write_to_postgres(df_clean, "fleet_data_cleaned")
    write_to_parquet(df_clean, "data/output/cleaned_fleet_data.parquet")

    # Cleanup
    spark.stop()
    print(f"✅ Batch ETL completed in {time.time() - start_time:.2f} sec")


# Entry point
if __name__ == "__main__":
    main()
