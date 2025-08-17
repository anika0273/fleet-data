"""
ml_anomaly.py - Production-ready fleet telemetry anomaly detection

Features:
1. Rule-based anomalies (overspeed, collisions, idle, battery)
2. Z-score statistical anomalies per vehicle
3. Clustering (KMeans) to detect unusual vehicle behavior
4. Modular ML plug-in design for future models
5. Writes results to PostgreSQL for dashboards and alerts

Run: spark-submit \
  --jars /Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar \
  metrics/ml_anomaly.py

Verify PostgreSQL table vehicle_anomalies_ml for results:

SELECT * 
FROM vehicle_anomalies_ml
ORDER BY window_end DESC, vehicle_id
LIMIT 50;

is_anomaly → True if any rule, z-score, or cluster anomaly is detected

anomaly_score → sum of all anomaly contributions

Individual columns (overspeed_anomaly, collision_anomaly, avg_speed_trimmed_z_anomaly, cluster_anomaly) are all interpretable
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lit, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# -----------------------------
# Config
# -----------------------------
POSTGRES_JDBC = "/Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar"
PG_URL = "jdbc:postgresql://localhost:5433/fleet_db"
PG_PROPS = {"user": "postgres", "password": "1234", "driver": "org.postgresql.Driver"}

INPUT_TABLE = "telemetry_metrics_vehicle_5m"
OUTPUT_TABLE = "vehicle_anomalies_ml"

# Rule thresholds
OVERSPEED_THRESHOLD = 0.2
COLLISION_THRESHOLD = 1
IDLE_RATIO_THRESHOLD = 0.8
BATTERY_LOW_THRESHOLD = 0.5

# Clustering
N_CLUSTERS = 4  # Number of typical vehicle behavior groups

# -----------------------------
# Spark session
# -----------------------------
def get_spark_session():
    return (
        SparkSession.builder
        .appName("FleetAnomalyDetection-ML")
        .config("spark.jars", POSTGRES_JDBC)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

# -----------------------------
# Load input metrics
# -----------------------------
def load_metrics(spark):
    df = spark.read.jdbc(PG_URL, INPUT_TABLE, properties=PG_PROPS)
    print(f"[INFO] Loaded {df.count()} rows from {INPUT_TABLE}")
    return df

# -----------------------------
# Rule-based anomaly detection
# -----------------------------
def rule_based_flags(df):
    df = df.withColumn("overspeed_anomaly", col("overspeed_rate") > lit(OVERSPEED_THRESHOLD))
    df = df.withColumn("collision_anomaly", col("collision_count") >= lit(COLLISION_THRESHOLD))
    df = df.withColumn("idle_anomaly", col("idle_ratio") > lit(IDLE_RATIO_THRESHOLD))
    df = df.withColumn("battery_anomaly", col("battery_low_ratio") > lit(BATTERY_LOW_THRESHOLD))
    return df

# -----------------------------
# Statistical anomaly detection (z-score)
# -----------------------------
def zscore_anomaly(df, features):
    w = Window.partitionBy("vehicle_id")
    for f in features:
        mean_col = F.mean(col(f)).over(w)
        std_col = F.stddev(col(f)).over(w)
        z_col = (col(f) - mean_col) / std_col
        df = df.withColumn(f"{f}_zscore", z_col)
        df = df.withColumn(f"{f}_z_anomaly", when(F.abs(z_col) > 3, True).otherwise(False))
    return df

# -----------------------------
# Clustering-based anomaly detection
# -----------------------------
def cluster_anomaly(df, features, n_clusters=N_CLUSTERS):
    """
    1. Vectorize features
    2. Standardize
    3. Apply KMeans clustering
    4. Compute distance from cluster centroid as anomaly score
    """
    assembler = VectorAssembler(inputCols=features, outputCol="features_vec")
    df_vec = assembler.transform(df)

    scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features")
    df_scaled = scaler.fit(df_vec).transform(df_vec)

    kmeans = KMeans(k=N_CLUSTERS, featuresCol="scaled_features", predictionCol="cluster")
    model = kmeans.fit(df_scaled)
    df_clustered = model.transform(df_scaled)

    # Distance to cluster center
    centers = model.clusterCenters()
    def distance_to_center(pred_col, features_col):
        from pyspark.sql.functions import udf
        from pyspark.ml.linalg import Vectors
        import numpy as np
        def dist(cluster_idx, vec):
            c = np.array(centers[cluster_idx])
            return float(np.linalg.norm(np.array(vec) - c))
        return udf(dist, "double")(col(pred_col), col(features_col))

    df_clustered = df_clustered.withColumn("cluster_distance", distance_to_center("cluster", "scaled_features"))

    # Flag as anomaly if distance > 95th percentile
    threshold = df_clustered.approxQuantile("cluster_distance", [0.95], 0.01)[0]
    df_clustered = df_clustered.withColumn("cluster_anomaly", col("cluster_distance") > lit(threshold))

    return df_clustered

# -----------------------------
# Aggregate anomaly scores
# -----------------------------
def compute_anomaly_score(df):
    rule_cols = ["overspeed_anomaly", "collision_anomaly", "idle_anomaly", "battery_anomaly"]
    z_cols = [c for c in df.columns if "_z_anomaly" in c]
    df = df.withColumn("rule_score", sum([col(c).cast("int") for c in rule_cols]))
    df = df.withColumn("zscore_score", sum([col(c).cast("int") for c in z_cols]))
    df = df.withColumn("cluster_score", col("cluster_anomaly").cast("int"))
    df = df.withColumn("anomaly_score", col("rule_score") + col("zscore_score") + col("cluster_score"))
    df = df.withColumn("is_anomaly", col("anomaly_score") > 0)
    return df

# -----------------------------
# Persist to Postgres
# -----------------------------
def write_to_postgres(df):
    df.write.mode("overwrite").jdbc(PG_URL, OUTPUT_TABLE, properties=PG_PROPS)
    print(f"[INFO] Wrote anomalies to {OUTPUT_TABLE}")

# -----------------------------
# Main execution
# -----------------------------
def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Load metrics
    df = load_metrics(spark)

    # Apply rule-based flags
    df = rule_based_flags(df)

    # Z-score anomaly detection
    features_for_zscore = ["avg_speed_trimmed", "speed_stddev", "overspeed_rate",
                           "braking_count", "harsh_accel_count", "collision_count"]
    df = zscore_anomaly(df, features_for_zscore)

    # Remove rows with nulls in cluster/zscore features
    df = df.dropna(subset=features_for_zscore)
    print(f"[INFO] Number of rows after dropna for clustering: {df.count()}")
    if df.count() == 0:
        print("[ERROR] No rows left for ML after dropna! Check your data for missing values.")
        spark.stop()
        return

    # Clustering-based anomaly detection
    df = cluster_anomaly(df, features_for_zscore)

    # Aggregate anomaly scores
    df = compute_anomaly_score(df)

    # Persist results
    write_to_postgres(df)

    spark.stop()

if __name__ == "__main__":
    main()
