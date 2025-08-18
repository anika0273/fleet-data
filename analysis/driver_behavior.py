# analysis/driver_behavior.py
# Driver Behavior Analysis (A/B Testing + ML)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from scipy import stats
import pandas as pd
import os

# Initialize Spark session
spark = SparkSession.builder.appName("DriverBehaviorAnalysis").getOrCreate()

def load_data(path):
    """
    Load cleaned telemetry data from batch ETL output.
    Expected columns:
        - driver_id
        - intervention (0=no, 1=yes)
        - risky_events
        - speeding_events, harsh_braking, fuel_efficiency
    """
    return spark.read.csv(path, header=True, inferSchema=True)

def calculate_fuel_efficiency(df):
    """Add a fuel efficiency column for analysis (L/km)."""
    df = df.withColumn(
        "fuel_efficiency_l_per_km",
        spark_round(col("fuel_consumption_liters") / col("distance_traveled_km"), 3)
    )
    return df

def perform_ab_test(df):
    """
    Compare risky events for drivers with/without intervention.
    Converts Spark DF to Pandas for t-test simplicity.
    """
    pdf = df.select("intervention", "risky_events").toPandas()
    control = pdf[pdf['intervention']==0]['risky_events']
    treatment = pdf[pdf['intervention']==1]['risky_events']

    t_stat, p_val = stats.ttest_ind(control, treatment)
    print(f"[Driver Behavior] T-test result: t={t_stat:.3f}, p={p_val:.3f}")
    return p_val

def train_predictive_model(df):
    """
    Predict which drivers are likely to have risky events.
    ML workflow:
    - Features: speed, harsh braking, fuel efficiency
    - Target: high-risk driver (binary)
    """
    pdf = df.select("speeding_events", "harsh_braking", "fuel_efficiency_l_per_km", "risky_events").toPandas()
    pdf['high_risk'] = pdf['risky_events'] > pdf['risky_events'].median()

    X = pdf[['speeding_events', 'harsh_braking', 'fuel_efficiency_l_per_km']]
    y = pdf['high_risk']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    acc = model.score(X, y)
    print(f"[Driver Behavior] ML Model Accuracy: {acc:.3f}")
    return model

if __name__ == "__main__":
    df = load_data("../data/processed/driver_behavior.csv")
    df = calculate_fuel_efficiency(df)
    perform_ab_test(df)
    train_predictive_model(df)
