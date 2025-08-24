"""
driver_behavior.py

Goal:
------
Analyze the effect of a driver behavior intervention (e.g., Safety Training or Alert System)
on risky driving events, and identify drivers most likely to benefit from further interventions.

Business Value:
---------------
- Reduce accidents
- Lower insurance premiums
- Improve public safety

Story / Workflow:
-----------------
1. Load cleaned fleet data (ETL already performed, ML-ready features exist)
2. Analyze risky events with respect to intervention (A/B test + hypothesis test)
3. Train a model to predict which drivers are likely to be risky and benefit from interventions
4. Save outputs for further analysis, visualization, and reporting

Data Source:
------------
- PostgreSQL table: fleet_data_cleaned
- Features already engineered in ETL (fuel efficiency, downtime flag, etc.)
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# -----------------------------
# PostgreSQL Connection
# -----------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "fleet_db",
    "user": "postgres",
    "password": "1234"
}

def get_connection():
    """Return SQLAlchemy engine connected to PostgreSQL"""
    url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)


def load_data(table: str = "fleet_data_cleaned") -> pd.DataFrame:
    """
    Load the cleaned fleet data from PostgreSQL.
    This is the starting point for all analysis.
    """
    engine = get_connection()
    df = pd.read_sql(f"SELECT * FROM {table};", engine)
    print(f"[INFO] Loaded {len(df)} rows from {table}")
    return df


# -----------------------------
# A/B Testing & Hypothesis Test
# -----------------------------
def analyze_intervention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare risky event rates between drivers with intervention vs. control group.
    Hypothesis Test:
        H0: Intervention does NOT reduce risky events
        H1: Intervention reduces risky events
    Method: Two-proportion z-test
    """
    df["risky_event"] = ((df["collision_alert"]) |
                         (df["harsh_acceleration_event"]) |
                         (df["braking_event"])).astype(int)

    intervention = df[df["intervention_active"] == True]
    control = df[df["intervention_active"] == False]

    # Average risky event rates
    rate_intervention = intervention["risky_event"].mean()
    rate_control = control["risky_event"].mean()
    n_intervention = len(intervention)
    n_control = len(control)

    # Proportion z-test
    pooled_prob = (intervention["risky_event"].sum() + control["risky_event"].sum()) / (n_intervention + n_control)
    se = np.sqrt(pooled_prob * (1 - pooled_prob) * (1/n_intervention + 1/n_control))
    z_score = (rate_intervention - rate_control) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    results = pd.DataFrame({
        "group": ["intervention", "control"],
        "avg_risky_rate": [rate_intervention, rate_control],
        "n": [n_intervention, n_control],
        "diff": rate_intervention - rate_control,
        "z_score": z_score,
        "p_value": p_value
    })

    print(f"[INFO] Intervention risky rate: {rate_intervention:.4f}")
    print(f"[INFO] Control risky rate: {rate_control:.4f}")
    print(f"[INFO] Z-score = {z_score:.3f}, p-value = {p_value:.4f}")
    print("[INFO] Hypothesis Test Interpretation:")
    if p_value < 0.05:
        print("✅ Significant reduction in risky events due to intervention")
    else:
        print("❌ No significant effect detected")

    return results


# -----------------------------
# Machine Learning: Predict Risky Drivers
# -----------------------------
def train_driver_risk_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a logistic regression model to predict risky drivers.
    Features used (can be expanded based on domain knowledge):
        - driver_experience_years
        - vehicle_age_years
        - fuel_efficiency_l_per_km
        - intervention_active
    Target:
        - risky_event
    """
    features = ["driver_experience_years", "vehicle_age_years", "fuel_efficiency_l_per_km", "intervention_active"]
    X = df[features]
    y = df["risky_event"]

    # Drop rows with any NaNs in features or target
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=500, class_weight='balanced') # for class imbalance
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    print(f"[INFO] Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("[INFO] Classification Report:")
    print(classification_report(y_test, y_pred))

    df_preds = X_test.copy()
    df_preds["actual_risky_event"] = y_test
    df_preds["predicted_risky_event"] = y_pred
    df_preds["predicted_prob"] = y_pred_proba

    return df_preds


# -----------------------------
# Save Results
# -----------------------------
def save_outputs(ab_results: pd.DataFrame, ml_results: pd.DataFrame, folder_name="driver_analysis_outputs"):
    """
    Save all outputs (CSV, Parquet, PostgreSQL)
    """
    os.makedirs(folder_name, exist_ok=True)

    # CSV
    ab_results.to_csv(os.path.join(folder_name, "ab_test_results.csv"), index=False)
    ml_results.to_csv(os.path.join(folder_name, "driver_risk_predictions.csv"), index=False)

    # Parquet
    ab_results.to_parquet(os.path.join(folder_name, "ab_test_results.parquet"), index=False)
    ml_results.to_parquet(os.path.join(folder_name, "driver_risk_predictions.parquet"), index=False)

    # PostgreSQL
    engine = get_connection()
    ab_results.to_sql("ab_test_results", engine, if_exists="replace", index=False)
    ml_results.to_sql("driver_risk_predictions", engine, if_exists="replace", index=False)

    print(f"[INFO] Saved results to folder '{folder_name}' and PostgreSQL.")



import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Visualization / Reporting
# -----------------------------
def plot_intervention_effect(df: pd.DataFrame, ab_results: pd.DataFrame, ml_results: pd.DataFrame, folder_name="driver_analysis_outputs"):
    """
    Generate visualizations for:
    1. Distribution of risky events
    2. Intervention vs Control risky event rates
    3. Predicted driver risk probabilities
    4. Logistic regression feature importance
    Saves plots to the outputs folder.
    """
    os.makedirs(folder_name, exist_ok=True)

    # 1. Distribution of risky events
    plt.figure(figsize=(8,5))
    sns.countplot(x="risky_event", data=df)
    plt.title("Distribution of Risky Events")
    plt.xlabel("Risky Event (0=Safe, 1=Risky)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "risk_event_distribution.png"))
    plt.close()

    # 2. Intervention vs Control risky event rates
    plt.figure(figsize=(8,5))
    sns.barplot(x="group", y="avg_risky_rate", data=ab_results)
    plt.title("Risky Event Rate: Intervention vs Control")
    plt.ylabel("Average Risky Event Rate")
    plt.xlabel("Group")
    for i, row in ab_results.iterrows():
        plt.text(i, row["avg_risky_rate"] + 0.005, f'{row["avg_risky_rate"]:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "intervention_vs_control.png"))
    plt.close()

    # 3. Predicted driver risk probabilities
    plt.figure(figsize=(10,5))
    sns.histplot(ml_results["predicted_prob"], bins=30, kde=True)
    plt.title("Predicted Driver Risk Probability Distribution")
    plt.xlabel("Predicted Probability of Risky Event")
    plt.ylabel("Count of Drivers")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "predicted_risk_prob_distribution.png"))
    plt.close()

    # 4. Feature importance from logistic regression (ML model)
    # Only works if we return model coefficients (we can extend train_driver_risk_model to return model)
    # For simplicity, we recompute here from logistic regression
    features = ["driver_experience_years", "vehicle_age_years", "fuel_efficiency_l_per_km", "intervention_active"]
    X = ml_results[features]
    y = ml_results["actual_risky_event"]

    # Refit logistic regression on full dataset for visualization of coefficients
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0]
    }).sort_values(by="coefficient", key=abs, ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x="coefficient", y="feature", data=coef_df, palette="viridis")
    plt.title("Logistic Regression Feature Importance")
    plt.xlabel("Coefficient (Magnitude = Importance)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "feature_importance.png"))
    plt.close()

    print(f"[INFO] Plots saved to folder '{folder_name}' for reporting and storytelling.")



# -----------------------------
# Main Script: Storytelling Workflow
# -----------------------------
if __name__ == "__main__":
    print("[INFO] Loading cleaned fleet data...")
    df = load_data()

    print("[INFO] Step 1: Analyze Intervention Effectiveness (A/B + Hypothesis Test)")
    ab_results = analyze_intervention(df)

    print("[INFO] Step 2: Train ML Model to Predict Risky Drivers")
    ml_results = train_driver_risk_model(df)

    print("[INFO] Step 3: Save Outputs for Reporting & Visualization")
    save_outputs(ab_results, ml_results)

    print("[INFO] Step 4: Generate Visualizations for Reporting")
    plot_intervention_effect(df, ab_results, ml_results)

    print("[INFO] Analysis complete ✅")
