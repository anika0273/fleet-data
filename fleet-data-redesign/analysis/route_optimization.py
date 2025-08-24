"""
route_optimization.py

Goal:
-----
Evaluate whether route optimization recommendations reduce fleet fuel consumption.

Business Value:
---------------
- Lower fuel costs → direct savings for the company
- Reduced carbon emissions → environmental impact
- Evidence for scaling route optimization across fleet

Techniques / Approach:
---------------------
1. Statistical Inference / A-B Testing:
   - Compare fuel consumption for vehicles on optimized routes vs. traditional routes.
   - Compute mean difference and confidence intervals.
   - Hypothesis testing to check if optimization reduces consumption.

2. Regression / ML Modeling:
   - Predict fuel consumption using telemetry features.
   - Identify which features most impact fuel efficiency (e.g., speed, idle time, vehicle type).

3. Visualization & Reporting:
   - Plot differences in fuel consumption.
   - Feature importance / regression coefficients.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

from sqlalchemy import create_engine

# ----------------------------
# Config: PostgreSQL
# ----------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "fleet_db",
    "user": "postgres",
    "password": "1234"
}

def get_connection():
    """Create PostgreSQL connection string for SQLAlchemy/Pandas."""
    url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

def load_data(table: str = "fleet_data_cleaned") -> pd.DataFrame:
    """Load fleet dataset from PostgreSQL into Pandas."""
    engine = get_connection()
    query = f"SELECT * FROM {table};"
    df = pd.read_sql(query, engine)
    return df


# -------------------------------
# Feature Engineering
# -------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that may provide additional predictive value.
    """
    # Basic derived features
    df["fuel_per_km"] = df["fuel_consumption_liters"] / (df["distance_traveled_km"] + 1e-3)  # Prevent /0
    df["idle_ratio"] = df["idle_time_minutes"] / (df["distance_traveled_km"] + 1e-3)
    if "timestamp" in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    # Interactions
    df["optimized_x_experience"] = df["optimized_route_flag"].astype(int) * df.get("driver_experience_years", 0)
    # If you have categorical features like vehicle_type:
    if "vehicle_type" in df.columns:
        df = pd.get_dummies(df, columns=["vehicle_type"], drop_first=True)
    return df


# -------------------------------
# Data Cleaning/Imputation
# -------------------------------
def clean_and_impute(df: pd.DataFrame, features: list, target: str) -> (pd.DataFrame, pd.Series):
    """
    Impute missing values in features and clean target to remove NaNs before modeling.
    """
    X = df[features]
    y = df[target]
    
    # Drop rows where target is NaN
    non_na_mask = y.notna()
    X = X.loc[non_na_mask]
    y = y.loc[non_na_mask]
    
    print("Null rows before imputation:\n", X.isnull().sum())
    
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    return X_imputed, y


# -------------------------------
# Statistical Testing (A/B)
# -------------------------------
def analyze_route_optimization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare mean fuel consumption for optimized vs. traditional routes.
    Returns a summary table with means, stddevs, counts, t-test result.
    """
    # Remove rows with missing target
    df = df.dropna(subset=['fuel_consumption_liters'])
    summary = df.groupby("optimized_route_flag")["fuel_consumption_liters"].agg(
        avg_fuel="mean",
        std_fuel="std",
        count="count"
    ).reset_index()

    # Statistical significance using Welch's t-test
    control = df[df["optimized_route_flag"]==False]["fuel_consumption_liters"]
    optimized = df[df["optimized_route_flag"]==True]["fuel_consumption_liters"]
    t_stat, p_value = stats.ttest_ind(control, optimized, equal_var=False, nan_policy='omit')

    summary["t_stat"] = t_stat
    summary["p_value"] = p_value

    print("=== Route Optimization Fuel Consumption Analysis ===")
    print(summary)
    print("Interpretation:")
    if p_value < 0.05:
        print("✅ Statistically significant: optimized routes reduce fuel consumption.")
    else:
        print("⚠️ No statistically significant difference found. Review group sizes, effect sizes, and potential confounders.")

    return summary


# -------------------------------
# Regression & ML Modeling
# -------------------------------
def train_fuel_model(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Predict fuel consumption using multiple algorithms. Outputs predictions and feature importance.
    """
    features = [
        "speed", "distance_traveled_km", "idle_time_minutes",
        "vehicle_age_years", "fuel_efficiency_l_per_km", "optimized_route_flag",
        "fuel_per_km", "idle_ratio", "optimized_x_experience"
    ]
    if "hour_of_day" in df.columns: features.append("hour_of_day")
    if "day_of_week" in df.columns: features.append("day_of_week")
    # Add any extra engineered features from dummies
    features += [col for col in df.columns if col.startswith("vehicle_type_")]

    target = "fuel_consumption_liters"
    X, y = clean_and_impute(df, features, target)

    # Split for model testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.4f}")
    print(f"Linear Regression R^2: {r2_score(y_test, y_pred_lr):.4f}")

    # Random Forest (handles nonlinearities, gives feature importance)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest MSE: {mean_squared_error(y_test, y_pred_rf):.4f}")
    print(f"Random Forest R^2: {r2_score(y_test, y_pred_rf):.4f}")

    # Surface most important predictors
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n=== Feature Importance (Random Forest) ===")
    print(feature_importance)

    # Return actual vs predicted for all plots and reviews
    predictions = X_test.copy()
    predictions["actual_fuel"] = y_test
    predictions["predicted_fuel"] = y_pred_rf
    return [predictions, feature_importance]

# -------------------------------
# Visualization / Reporting
# -------------------------------
def plot_route_analysis(predictions: pd.DataFrame, feature_importance: pd.DataFrame, ab_results: pd.DataFrame, folder_name="route_analysis_outputs"):
    """
    Save plots:
    1. Groupwise bar (A/B comparison)
    2. Predicted vs actual (scatter)
    3. Feature importance
    """
    os.makedirs(folder_name, exist_ok=True)

    # Fuel consumption per group
    plt.figure(figsize=(8,5))
    sns.barplot(x="optimized_route_flag", y="avg_fuel", data=ab_results)
    plt.title("Average Fuel Consumption: Optimized vs Control")
    plt.ylabel("Fuel Consumption (Liters)")
    plt.xlabel("Route Optimization Applied")
    plt.xticks([0,1], ["Control", "Optimized"])
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "ab_fuel_comparison.png"))
    plt.close()

    # Predicted vs actual
    plt.figure(figsize=(10,5))
    sns.scatterplot(x="actual_fuel", y="predicted_fuel", data=predictions)
    plt.plot(
        [predictions["actual_fuel"].min(), predictions["actual_fuel"].max()],
        [predictions["actual_fuel"].min(), predictions["actual_fuel"].max()],
        color="red", linestyle="--"
    )
    plt.title("Predicted vs Actual Fuel Consumption (Random Forest)")
    plt.xlabel("Actual Fuel Consumption")
    plt.ylabel("Predicted Fuel Consumption")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "predicted_vs_actual_fuel.png"))
    plt.close()

    # Feature importance
    plt.figure(figsize=(8,5))
    sns.barplot(x="importance", y="feature", data=feature_importance.head(12), palette="viridis")
    plt.title("Feature Importance for Fuel Consumption Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "feature_importance.png"))
    plt.close()
    print(f"[INFO] Plots saved to folder '{folder_name}' for reporting.")

# -------------------------------
# Save outputs
# -------------------------------
def save_outputs(predictions, feature_importance, folder_name="route_analysis_outputs"):
    os.makedirs(folder_name, exist_ok=True)
    predictions.to_csv(os.path.join(folder_name, "predicted_fuel.csv"), index=False)
    predictions.to_parquet(os.path.join(folder_name, "predicted_fuel.parquet"), index=False)
    feature_importance.to_csv(os.path.join(folder_name, "feature_importance.csv"), index=False)
    print(f"[INFO] Outputs saved to '{folder_name}'.")

# -------------------------------
# Main Script
# -------------------------------
def main():
    print("[INFO] Step 1: Load data from PostgreSQL")
    df = load_data()
    print(f"[INFO] Loaded {len(df)} rows.")

    print("[INFO] Step 2: Feature engineering")
    df = feature_engineering(df)

    print("[INFO] Step 3: Statistical Analysis / A-B Testing")
    ab_results = analyze_route_optimization(df)

    print("[INFO] Step 4: Regression / ML Modeling")
    results = train_fuel_model(df)
    predictions = results[0]
    feature_importance = results[1]


    print("[INFO] Step 5: Visualization / Reporting")
    plot_route_analysis(predictions, feature_importance, ab_results)

    print("[INFO] Step 6: Save outputs for further analysis")
    save_outputs(predictions, feature_importance)

    print("[INFO] ✅ Route Optimization Analysis Completed!")

# Entry point
if __name__ == "__main__":
    main()