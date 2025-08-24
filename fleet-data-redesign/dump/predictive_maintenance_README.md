# Predictive Maintenance Analysis for Fleet Data

## Overview

This project analyzes the impact of Proactive Predictive Maintenance compared to Scheduled Maintenance on fleet vehicle downtime and breakdown events. It also develops machine learning models to predict vehicle breakdown risk using telemetry and maintenance data.

---

## Objectives

1. **Statistical Comparison:**  
   Evaluate whether predictive maintenance reduces unplanned vehicle downtime and breakdown rates compared to scheduled maintenance, using vehicle-level and record-level statistical analysis.

2. **Predictive Modeling:**  
   Build models to predict breakdown events leveraging telemetry and device data, incorporating methods to handle class imbalance and improve prediction reliability.

3. **Business Impact Estimation:**  
   Translate observed maintenance effects into estimated downtime savings to support decision making.

---

## Data Source

- Dataset: `fleet_data_cleaned` table in PostgreSQL, generated from a Spark ETL pipeline.
- Each row corresponds to a vehicle trip or snapshot and contains:
  - `maintenance_type`: either 'predictive' or 'scheduled'
  - `downtime_hours`: numeric, representing hours lost
  - `breakdown_event`: boolean label for breakdown occurrence
  - Telemetry/device features: e.g., speed, idle time, device generation, sensor battery, GPS accuracy, etc.
- Dataset size: ~99,000 rows with significant class imbalance (~1% positive breakdown events).

---

## Analysis Workflow

### 1. Data Loading & Cleaning

- Load dataset from PostgreSQL.
- Convert numerical columns to proper numeric types.
- Filter out rows missing critical maintenance type or downtime values.
- Prepare binary labels and features for analysis.

### 2. Statistical Tests (Hypothesis Testing)

- Aggregate data to vehicle-level to reduce volume bias.
- Compute means and breakdown rates per maintenance type.
- Perform Welch’s t-test and Mann–Whitney U test to compare groups on:
  - Mean downtime (hours)
  - Breakdown event rates
- Compute 95% confidence intervals for mean differences.
- Perform record-level tests as sanity checks.
- Save test summaries to CSV, Parquet, and PostgreSQL.

### 3. Feature Engineering for Modeling

- Select telemetry, device health, and maintenance-related features including:
  - `vehicle_age_years`, `device_generation`, `time_since_last_maintenance_days`
  - `speed`, `idle_time_minutes`, `data_latency_ms`, `gps_accuracy_meters`, `packet_loss_rate`, `sensor_battery`
  - `rush_hour` and `maintenance_type` (binary encoded)
- Create new interaction feature: `maintenance_type * vehicle_age_years`.
- Drop samples with missing values in selected features.

### 4. Handling Class Imbalance

- Use SMOTE (Synthetic Minority Oversampling Technique) applied only on the training set to oversample minority breakdown events and balance classes.

### 5. Model Training and Evaluation

- Split data into train/test subsets with stratified sampling to preserve class distribution.
- Train four classifiers:
  - Logistic Regression with class weights and calibrated probabilities.
  - Random Forest with class weight balancing.
  - XGBoost classifier with scale_pos_weight adjustment.
  - LightGBM classifier trained with validation set and early stopping via callback.
- Evaluate models with:
  - ROC-AUC and Precision-Recall AUC metrics.
  - Confusion matrices at F1-score maximizing thresholds selected from precision-recall curves.
- Extract Random Forest feature importance and Logistic Regression coefficients for interpretability.
- Save model metrics, feature importances, coefficients, and predictions to CSV, Parquet, and PostgreSQL.

### 6. Visualization & Reporting

- Generate plots for:
  - Downtime by maintenance type (boxplots).
  - Breakdown rates by maintenance type (bar charts).
  - ROC and Precision-Recall curves comparing all models.
  - Feature importance from Random Forest.
- Save plots for reporting and embed insights into logs.

### 7. Business Impact Estimation

- Use difference in mean downtime between maintenance types to estimate hours saved per 1000 vehicle records.
- Provide confidence intervals and interpret findings to quantify potential operational savings.

---

## Results Summary

### Statistical Findings

- Predictive maintenance vehicles exhibit slightly lower average downtime than scheduled, but difference is _not statistically significant_ (p=0.125).
- Breakdown rates are significantly lower in predictive maintenance vehicles (p=2.36e-05), indicating a potential reduction in failure likelihood despite small numeric difference.
- Estimated downtime savings of approximately 18 hours per 1000 vehicle records switching from scheduled to predictive maintenance.

### Model Performance

- Models achieve ROC-AUC scores close to random chance (~0.51–0.55), indicating weak predictive signals likely due to limited or noisy synthetic features and strong class imbalance.
- Precision-Recall AUC and F1-scores are very low, reflecting low sensitivity in predicting rare breakdown events.
- LightGBM performs marginally better than others but still limited.
- Feature importance suggests device age, maintenance recency, and telemetry data contribute to risk prediction.

---

## Interpretation & Limitations

- Results reflect challenges typical in predictive maintenance datasets: highly imbalanced targets and complex failure patterns.
- Synthetic data likely lacks rich, causally predictive structure found in real-world fleets, limiting model learning.
- Statistical significance in breakdown reduction suggests potential real effect; however, stronger models or richer data required to make accurate individual predictions.
- Business impact estimates provide useful order-of-magnitude savings but should be cross-validated with operational data.

---

## Recommendations for Future Work

1. **Data Improvements**

   - Incorporate richer temporal/sequence data (histories of failures, sensor trends).
   - Add driver, route, and environmental context features.
   - Use real-world fleet data if possible to capture complex failure drivers.

2. **Feature Engineering Enhancements**

   - Derive rolling statistics and lags.
   - Add non-linear interactions and domain-specific metrics.

3. **Advanced Modeling**

   - Explore ensemble and deep learning models tailored for rare event prediction.
   - Apply cost-sensitive and anomaly detection approaches.
   - Use time-series or survival analysis models for failure prediction.

4. **Model Evaluation**

   - Employ time-based validation splits to prevent leakage.
   - Use calibration plots and decision curve analysis.

5. **Explainability & Monitoring**
   - Employ SHAP or LIME for deeper interpretation.
   - Establish monitoring pipelines to track predictive maintenance impact over time.

---

## Conclusion

This analysis demonstrates a complete workflow from statistical testing through advanced ML modeling for predictive maintenance evaluation. While predictive power on synthetic data is limited, the framework and insights establish a strong foundation for application to real-world fleet data and ongoing model refinement.
