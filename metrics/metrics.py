"""
Advanced Fleet Metrics Calculation (Section 5)
---------------------------------------------

Purpose:
- Compute real-world rolling KPIs per vehicle using a 5-minute window with 1-minute slide.
- Metrics cover speed, overspeed, safety events, operational ratios, and data quality.
- Supports streaming or backfilled static data from PostgreSQL.
- Writes metrics to telemetry_metrics_vehicle_5m table.

Source Tables:
1) fleet_stream_processed   (preferred: latest streaming snapshot)
2) fleet_data_cleaned       (fallback: cleaned historical data)

Output Table:
- telemetry_metrics_vehicle_5m

Notes:
- This module is designed to mirror real fleet ops monitoring dashboards.
- KPIs are actionable: trigger alerts, build ML features, track compliance.

Execution:
spark-submit --jars /path/to/postgresql-42.7.3.jar metrics/metrics.py
"""

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, when, window, avg, count, sum as fsum, stddev_samp,
    percentile_approx, lit
)

# -----------------------------
# Config
# -----------------------------
POSTGRES_JDBC = "/Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar"
PG_URL = "jdbc:postgresql://localhost:5433/fleet_db"
PG_PROPS = {"user": "postgres", "password": "1234", "driver": "org.postgresql.Driver"}

# Output table for metrics
OUT_TABLE_VEHICLE_5M = "telemetry_metrics_vehicle_5m"

# Road-type speed limits (km/h) and overspeed tolerance
SPEED_LIMITS = {"city": 50.0, "highway": 110.0, "rural": 90.0}
OVERSPEED_TOLERANCE = 0.10  # +10% above nominal limit

# -----------------------------
# Spark Session
# -----------------------------
def get_spark_session():
    """
    Create SparkSession with PostgreSQL JDBC and minimal local partitions.
    """
    return (
        SparkSession.builder
        .appName("FleetMetricsJob-Advanced")
        .config("spark.jars", POSTGRES_JDBC)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

# -----------------------------
# Load Best Available Input
# -----------------------------
def load_best_available_input(spark):
    """
    Prefer streaming table; fallback to cleaned static table.
    """
    candidates = ["fleet_stream_processed", "fleet_data_cleaned"]
    for tbl in candidates:
        try:
            df = spark.read.jdbc(PG_URL, tbl, properties=PG_PROPS)
            if df.take(1):
                print(f"[INFO] Using source table: {tbl}")
                return df
        except Exception as e:
            print(f"[WARN] Could not read {tbl}: {e}")
    raise RuntimeError("No suitable input table found.")

# -----------------------------
# Data Enrichment & Flags
# -----------------------------
def robustify_and_flag(df):
    """
    1) Create operational and safety flags
    2) Winsorize speed to remove extreme outliers
    3) Compute overspeed flags
    4) Data quality flags
    """
    # Data quality
    df = df.withColumn("gps_missing", col("latitude").isNull() | col("longitude").isNull())
    df = df.withColumn("battery_low", col("sensor_battery").isNotNull() & (col("sensor_battery") < 20.0))
    df = df.withColumn("idle_flag", col("speed").isNotNull() & (col("speed") <= 1.0))

    # Safety flags
    df = df.withColumn("braking_flag", col("braking_event") == True)
    df = df.withColumn("harsh_accel_flag", col("harsh_acceleration_event") == True)
    df = df.withColumn("collision_flag", col("collision_alert") == True)

    # Winsorize speed (1st-99th percentile)
    bounds = df.select(
        percentile_approx("speed", 0.01, 1000).alias("p01"),
        percentile_approx("speed", 0.99, 1000).alias("p99"),
    ).collect()[0]
    lo, hi = bounds["p01"], bounds["p99"]
    df = df.withColumn(
        "speed_trimmed",
        when(col("speed").isNull(), None)
        .when(col("speed") < lit(lo), lit(lo))
        .when(col("speed") > lit(hi), lit(hi))
        .otherwise(col("speed"))
    )

    # Overspeed flag
    limit_city = lit(SPEED_LIMITS.get("city", 50.0))
    limit_highway = lit(SPEED_LIMITS.get("highway", 110.0))
    limit_rural = lit(SPEED_LIMITS.get("rural", 90.0))
    base_limit = (
        when(col("road_type") == "city", limit_city)
        .when(col("road_type") == "highway", limit_highway)
        .when(col("road_type") == "rural", limit_rural)
        .otherwise(lit(80.0))
    )
    threshold = base_limit * (1.0 + lit(OVERSPEED_TOLERANCE))
    df = df.withColumn("overspeed_flag", col("speed_trimmed") > threshold)

    return df

# -----------------------------
# Windowed Metrics Calculation
# -----------------------------
def aggregate_vehicle_window_metrics(df):
    """
    Compute per-vehicle, 5-minute window metrics (sliding 1 minute):
    - avg_speed_trimmed, speed_stddev
    - overspeed_rate
    - braking/harsh_accel/collision counts
    - idle_ratio, battery_low_ratio, gps_missing_ratio
    """
    win = window(col("timestamp"), "5 minutes", "1 minute") # Window length: 5 minutes → every metric aggregates events in a 5-minute interval. Window slide: 1 minute → a new window starts every minute. This means each event can appear in multiple overlapping windows.
    base_group = df.groupBy("vehicle_id", win)

    def sum_bool(flag_col):
        return fsum(when(col(flag_col) == True, 1).otherwise(0))

    metrics = base_group.agg(
        avg("speed_trimmed").alias("avg_speed_trimmed"),
        stddev_samp("speed_trimmed").alias("speed_stddev"),
        sum_bool("braking_flag").alias("braking_count"),
        sum_bool("harsh_accel_flag").alias("harsh_accel_count"),
        sum_bool("collision_flag").alias("collision_count"),
        sum_bool("overspeed_flag").alias("overspeed_events"),
        sum_bool("idle_flag").alias("idle_events"),
        sum_bool("battery_low").alias("battery_low_events"),
        sum_bool("gps_missing").alias("gps_missing_events"),
        count("*").alias("row_count")
    )

    # Convert counts to rates; guard against divide-by-zero
    metrics = metrics.select(
        col("vehicle_id"),
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("avg_speed_trimmed"),
        col("speed_stddev"),
        col("braking_count"),
        col("harsh_accel_count"),
        col("collision_count"),
        (col("overspeed_events") / F.when(col("row_count") > 0, col("row_count")).otherwise(lit(1))).alias("overspeed_rate"),
        (col("idle_events") / F.when(col("row_count") > 0, col("row_count")).otherwise(lit(1))).alias("idle_ratio"),
        (col("battery_low_events") / F.when(col("row_count") > 0, col("row_count")).otherwise(lit(1))).alias("battery_low_ratio"),
        (col("gps_missing_events") / F.when(col("row_count") > 0, col("row_count")).otherwise(lit(1))).alias("gps_missing_ratio"),
        col("row_count")
    )

    return metrics

# -----------------------------
# Main
# -----------------------------
def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Load source table (streaming preferred)
    raw = load_best_available_input(spark)

    # Deduplicate by vehicle_id + timestamp
    dedupe_w = Window.partitionBy("vehicle_id", "timestamp").orderBy(col("timestamp").desc())
    raw = raw.withColumn("rn", F.row_number().over(dedupe_w)).filter(col("rn") == 1).drop("rn")

    # Enrich flags & robust speed
    enriched = robustify_and_flag(raw)

    # Compute 5-minute sliding metrics
    vehicle_5m = aggregate_vehicle_window_metrics(enriched)

    # Persist to PostgreSQL
    vehicle_5m.write.mode("append").jdbc(PG_URL, OUT_TABLE_VEHICLE_5M, properties=PG_PROPS)

    print(f"[OK] Wrote vehicle window metrics to {OUT_TABLE_VEHICLE_5M}")
    spark.stop()


if __name__ == "__main__":
    main()


# Trimmed avg speed (avg_speed_trimmed): normalizes away 1% extreme outliers (e.g., GPS glitches) while preserving genuine highs/lows. Gives a stable “typical” speed per vehicle & window. \
# Speed volatility (speed_stddev): high volatility correlates with aggressive driving or dense stop-and-go traffic. Useful for driver coaching and ETA reliability. \ 
# Overspeed rate (overspeed_rate): % of rows exceeding a road-type specific limit with tolerance. This is a genuine policy compliance KPI and safety risk indicator.\ 
# Braking / harsh accel / collision counts: raw intensity of risky behaviors in the window—perfect for triggers and incident triage. \
# Idle ratio (idle_ratio): time spent at or near zero speed. Great for utilization, fuel, and “why isn’t this asset moving?” investigations. \
# Battery low ratio: proactive maintenance for edge devices (replace/recharge before they go dark). \
# GPS missing ratio: data quality SLA; high values mean you can’t trust location-based analytics for that vehicle/window.