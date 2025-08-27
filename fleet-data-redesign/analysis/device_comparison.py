"""
device_comparison.py

Objective (Story)
-----------------
5) Are vehicles using a “new telematics device” more efficient and reliable
   than those on the previous generation?

Business Value
--------------
- Validate ROI of device upgrades (fuel savings, uptime, fewer safety events).
- Support rollout decisions and vendor negotiations with evidence.
- Identify contexts where the new device brings the largest gains.

Data Source
-----------
- Static table: `fleet_data_cleaned` in PostgreSQL (from Spark ETL).
- Key fields:
    * device_generation (int) .............. proxy for "new vs old" hardware
    * fuel_rate_l_per_100km (float) ........ efficiency (↓ better)
    * downtime_hours (float) ............... reliability (↓ better)
    * maintenance_cost_usd (float) ......... reliability/wear proxy (↓ better)
    * breakdown_event (bool) ............... reliability (↓ better)
    * safety events: harsh_acceleration_event, braking_event, collision_alert (↓ better)
    * device metrics: latency, gps accuracy, packet loss, faults (↓ better)
    * context covariates: road_type, traffic_density, weather, hour_of_day, rush_hour,
                          driver_experience_years, driver_training, vehicle_age_years, vehicle_type, etc.

Approach (What & Why)
---------------------
A) **A/B + Hypothesis Testing**
   - Define "new device" vs "old device" using a configurable threshold (default >= 2 is "new").
   - Compare KPI means (fuel_rate, downtime, maintenance cost, latency) via Welch t-test
     and robust Mann–Whitney. Compare event rates (breakdown, collision, harsh/braking/sensor faults)
     via chi-square tests. This answers: "Is there a statistically significant improvement?"

B) **Causal Adjustment (Propensity Score Weighting)**
   - New devices might be deployed to newer vehicles, certain depots, or on premium routes—bias!
   - Estimate P(New | X) using covariates (pre-treatment context & characteristics).
   - Compute stabilized inverse-propensity weights to estimate ATE and bootstrap CIs.
   - Report weighted differences on fuel_rate, downtime, cost, breakdown_event.

C) **Predict "Adoption Effect" / Who Benefits Most (T-Learner)**
   - For *continuous* efficiency: model E[fuel_rate | X, T=0] and E[fuel_rate | X, T=1]
     using RandomForestRegressor → uplift = y_old - y_new (expected reduction in fuel_rate).
   - For *binary* reliability: model P(breakdown | X, T=0/1) via RandomForestClassifier
     → uplift_prob = p0 - p1 (expected reduction in breakdown probability).
   - Aggregate by vehicle_id to give a ranked “upgrade targeting” list.
   - Train a meta-model on predicted uplift to extract which features *modulate*
     the benefit (heuristic “what features most benefit from new tech”).

Outputs
-------
Saved under: fleet-data/analysis_outputs/device_comparison/
    * CSV + Parquet:
        - dev_ab_summary.(csv|parquet)
        - dev_ipw_summary.(csv|parquet)
        - dev_uplift_by_vehicle.(csv|parquet)
        - dev_uplift_deciles.(csv|parquet)
        - dev_uplift_feature_modulators.(csv|parquet)
        - dev_model_details.(csv|parquet)
    * PostgreSQL (if_exists="replace"):
        - device_ab_summary
        - device_ipw_summary
        - device_uplift_by_vehicle
        - device_uplift_deciles
        - device_uplift_feature_modulators
        - device_model_details
    * Figures (PNG):
        - fuel_rate_by_device.png
        - breakdown_rate_by_device.png
        - downtime_by_device.png
        - uplift_deciles_fuel_rate.png
        - uplift_deciles_breakdown.png
        - propensity_hist_new_device.png

How to Run
----------
docker compose up postgres
docker compose up device_comparison_analysis
"""

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, log_loss, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# =============================
# Config
# =============================
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # enable project-root imports

from config import config  # now you can use config.POSTGRES_HOST, etc.

DB_CONFIG = {
    "host": config.POSTGRES_HOST,
    "port": config.POSTGRES_PORT,
    "database": config.POSTGRES_DB,
    "user": config.POSTGRES_USER,
    "password": config.POSTGRES_PASSWORD,
}

# Threshold that defines "new device"
NEW_DEVICE_THRESHOLD = 2  # device_generation >= 2 → New; 0/1 → Old (adjust as needed)

# Output folder
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "device_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# DB Helpers & IO
# =============================
def get_connection():
    url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

def load_data(table: str = "fleet_data_cleaned") -> pd.DataFrame:
    engine = get_connection()
    return pd.read_sql(f"SELECT * FROM {table};", engine)

def df_to_files(df: pd.DataFrame, name: str):
    df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
    df.to_parquet(os.path.join(OUTPUT_DIR, f"{name}.parquet"), index=False)

def df_to_postgres(df: pd.DataFrame, table_name: str, if_exists="replace"):
    eng = get_connection()
    df.to_sql(table_name, eng, index=False, if_exists=if_exists)

# =============================
# Preparation
# =============================
def add_device_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'new_device' flag based on device_generation.
    Rationale:
      - A clean, binary split for A/B, causal adjustments, and uplift learning.
      - Keep original integer device_generation for descriptive stats or sensitivity.
    """
    df = df.copy()
    df["new_device"] = (df["device_generation"] >= NEW_DEVICE_THRESHOLD).astype(bool)
    return df

# =============================
# A/B + Hypothesis Testing
# =============================
def device_ab_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare new_device vs old_device across KPIs.
    - Proportions (chi-square): breakdown_event, collision_alert,
      harsh_acceleration_event, braking_event, sensor_fault_event, gps_loss_event, network_delay_event
    - Means (Welch + Mann–Whitney): fuel_rate_l_per_100km, downtime_hours,
      maintenance_cost_usd, data_latency_ms, gps_accuracy_meters, packet_loss_rate
    """
    # Type safety
    bool_cols = [
        "breakdown_event", "collision_alert",
        "harsh_acceleration_event", "braking_event",
        "sensor_fault_event", "gps_loss_event", "network_delay_event",
        "new_device"
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    treat = df[df["new_device"]]
    ctrl  = df[~df["new_device"]]

    rows = []

    # --- Proportion tests
    prop_cols = [
        "breakdown_event", "collision_alert",
        "harsh_acceleration_event", "braking_event",
        "sensor_fault_event", "gps_loss_event", "network_delay_event"
    ]
    for col in prop_cols:
        if col not in df.columns:
            continue
        tab = pd.crosstab(df["new_device"], df[col])
        if tab.shape == (2,2):
            chi2, p_chi, _, _ = stats.chi2_contingency(tab.values)
        else:
            chi2, p_chi = np.nan, np.nan
        m_t = treat[col].mean()
        m_c = ctrl[col].mean()
        rows.append({
            "metric": f"{col}_rate",
            "treatment_mean": m_t,
            "control_mean": m_c,
            "diff_t_minus_c": m_t - m_c,
            "test": "chi2",
            "stat": chi2, "p_value": p_chi,
            "n_treat": len(treat), "n_control": len(ctrl)
        })

    # --- Mean tests
    mean_cols = [
        "fuel_rate_l_per_100km", "downtime_hours",
        "maintenance_cost_usd", "data_latency_ms",
        "gps_accuracy_meters", "packet_loss_rate"
    ]
    for col in mean_cols:
        if col not in df.columns:
            continue
        t_stat, p_t = stats.ttest_ind(treat[col], ctrl[col], equal_var=False, nan_policy="omit")
        try:
            u_stat, p_u = stats.mannwhitneyu(treat[col], ctrl[col], alternative="two-sided")
        except ValueError:
            u_stat, p_u = np.nan, np.nan
        rows.append({
            "metric": f"{col}_mean",
            "treatment_mean": treat[col].mean(),
            "control_mean": ctrl[col].mean(),
            "diff_t_minus_c": treat[col].mean() - ctrl[col].mean(),
            "test": "welch_t",
            "stat": t_stat, "p_value": p_t,
            "mw_u": u_stat, "mw_p": p_u,
            "n_treat": len(treat), "n_control": len(ctrl)
        })

    return pd.DataFrame(rows)

# =============================
# Propensity Score Weighting
# =============================
def ipw_new_device(df: pd.DataFrame, seed=42, bootstrap_iters=200) -> pd.DataFrame:
    """
    Stabilized IPW to estimate ATE of 'new_device' on key outcomes.
    Covariates exclude post-treatment device telemetry (e.g., latency, packet loss).
    """
    covars_cat = ["road_type", "traffic_density", "weather", "vehicle_type", "driver_training"]
    covars_num = ["hour_of_day", "rush_hour", "driver_experience_years",
                  "vehicle_age_years", "speed", "idle_time_minutes",
                  "distance_traveled_km", "fuel_rate_l_per_100km",  # NOTE: careful if including outcome-like covariates
                  "time_since_last_maintenance_days"]
    # Depending on assumptions, you may want to DROP 'fuel_rate_l_per_100km' from covariates to avoid adjusting away treatment effect.
    # We keep it here commented for transparency. Safer default: remove it from covariates.
    covars_num = [c for c in covars_num if c != "fuel_rate_l_per_100km"]

    outcomes = [
        ("fuel_rate_l_per_100km", "mean"),
        ("downtime_hours", "mean"),
        ("maintenance_cost_usd", "mean"),
        ("breakdown_event", "proportion"),
    ]
    keep = ["new_device"] + covars_cat + covars_num + [y for y, _ in outcomes]
    d = df[keep].dropna().copy()
    d["T"] = d["new_device"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), covars_cat),
            ("num", "passthrough", covars_num)
        ],
        remainder="drop"
    )
    prop_model = Pipeline(steps=[
        ("prep", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    X, y = d[covars_cat + covars_num], d["T"]
    prop_model.fit(X, y)
    p = np.clip(prop_model.predict_proba(X)[:,1], 1e-3, 1-1e-3)
    pbar = y.mean()
    d["sw"] = np.where(d["T"]==1, pbar/p, (1-pbar)/(1-p))

    def wt_diff(y_col, dframe, kind):
        if kind == "proportion":
            # treat/control weighted means of binary outcome
            m_t = np.average(dframe.loc[dframe["T"]==1, y_col], weights=dframe.loc[dframe["T"]==1, "sw"])
            m_c = np.average(dframe.loc[dframe["T"]==0, y_col], weights=dframe.loc[dframe["T"]==0, "sw"])
        else:
            m_t = np.average(dframe.loc[dframe["T"]==1, y_col], weights=dframe.loc[dframe["T"]==1, "sw"])
            m_c = np.average(dframe.loc[dframe["T"]==0, y_col], weights=dframe.loc[dframe["T"]==0, "sw"])
        return m_t, m_c, m_t - m_c

    rows = []
    rng = np.random.default_rng(seed)
    for y_col, kind in outcomes:
        m_t, m_c, diff = wt_diff(y_col, d, kind)
        # bootstrap CI
        diffs = []
        for _ in range(bootstrap_iters):
            samp = d.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
            _, _, di = wt_diff(y_col, samp, kind)
            diffs.append(di)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({
            "outcome": y_col,
            "estimand": "ATE (stabilized IPW)",
            "treatment_wt_mean": m_t,
            "control_wt_mean": m_c,
            "diff_t_minus_c": diff,
            "ci95_low": lo, "ci95_high": hi,
            "n_used": len(d)
        })

    # Diagnostics: propensity histogram
    plt.figure(figsize=(7,5))
    sns.histplot(p, bins=30, kde=True)
    plt.title("Propensity P(New Device | X)")
    plt.xlabel("Propensity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "propensity_hist_new_device.png"))
    plt.close()

    return pd.DataFrame(rows)

# =============================
# T-Learner: Adoption Effect
# =============================
def t_learner_adoption_effect(df: pd.DataFrame, seed=42):
    """
    Estimate per-trip uplift from upgrading device:
      - Continuous efficiency: fuel_rate_l_per_100km → RF Regressor models
      - Binary reliability: breakdown_event → RF Classifier models

    Uplift definitions:
      fuel_uplift = E[fuel_rate|Old] - E[fuel_rate|New]  (positive = expected reduction)
      breakdown_uplift = P(breakdown|Old) - P(breakdown|New) (positive = expected reduction)

    Aggregate by vehicle_id for a ranked rollout plan.
    Also fit a meta-model on uplift to estimate which features *modulate* benefit.
    """
    # Features (avoid device metrics that are downstream of treatment)
    cats = ["road_type", "traffic_density", "weather", "vehicle_type", "driver_training"]
    nums = ["hour_of_day", "rush_hour", "driver_experience_years",
            "vehicle_age_years", "speed", "idle_time_minutes",
            "distance_traveled_km", "time_since_last_maintenance_days"]
    base_cols = ["vehicle_id", "device_generation", "new_device"] + cats + nums

    needed = base_cols + ["fuel_rate_l_per_100km", "breakdown_event"]
    d = df[needed].dropna().copy()
    d["T"] = d["new_device"].astype(int)
    d["Y_fuel"] = d["fuel_rate_l_per_100km"].astype(float)
    d["Y_break"] = d["breakdown_event"].astype(int)

    train_idx, test_idx = train_test_split(
        d.index, test_size=0.25, random_state=seed, stratify=d[["T", "Y_break"]]
    )
    tr, te = d.loc[train_idx].copy(), d.loc[test_idx].copy()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ("num", "passthrough", nums)
        ]
    )

    # --- Fuel rate models (Regressors)
    reg_old = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=-1))
    ])
    reg_new = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=-1))
    ])
    tr_old = tr[tr["T"]==0]
    tr_new = tr[tr["T"]==1]
    reg_old.fit(tr_old[cats+nums], tr_old["Y_fuel"])
    reg_new.fit(tr_new[cats+nums], tr_new["Y_fuel"])

    # Evaluate (sanity): MAE/R2 on each segment
    te_old = te[te["T"]==0]
    te_new = te[te["T"]==1]
    mae_old = mean_absolute_error(te_old["Y_fuel"], reg_old.predict(te_old[cats+nums])) if len(te_old) else np.nan
    mae_new = mean_absolute_error(te_new["Y_fuel"], reg_new.predict(te_new[cats+nums])) if len(te_new) else np.nan
    r2_old  = r2_score(te_old["Y_fuel"], reg_old.predict(te_old[cats+nums])) if len(te_old) else np.nan
    r2_new  = r2_score(te_new["Y_fuel"], reg_new.predict(te_new[cats+nums])) if len(te_new) else np.nan

    # Uplift on test
    p_fuel_old = reg_old.predict(te[cats+nums])
    p_fuel_new = reg_new.predict(te[cats+nums])
    te["fuel_uplift"] = p_fuel_old - p_fuel_new  # positive = reduction (good)

    # --- Breakdown models (Classifiers)
    clf_old = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1))
    ])
    clf_new = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestClassifier(n_estimators=350, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1))
    ])
    clf_old.fit(tr_old[cats+nums], tr_old["Y_break"])
    clf_new.fit(tr_new[cats+nums], tr_new["Y_break"])

    # Sanity metrics
    auc_old = roc_auc_score(te_old["Y_break"], clf_old.predict_proba(te_old[cats+nums])[:,1]) if len(te_old) else np.nan
    auc_new = roc_auc_score(te_new["Y_break"], clf_new.predict_proba(te_new[cats+nums])[:,1]) if len(te_new) else np.nan
    ll_old  = log_loss(te_old["Y_break"], clf_old.predict_proba(te_old[cats+nums])[:,1], labels=[0,1]) if len(te_old) else np.nan
    ll_new  = log_loss(te_new["Y_break"], clf_new.predict_proba(te_new[cats+nums])[:,1], labels=[0,1]) if len(te_new) else np.nan

    # Uplift probabilities (positive = expected reduction)
    p_break_old = clf_old.predict_proba(te[cats+nums])[:,1]
    p_break_new = clf_new.predict_proba(te[cats+nums])[:,1]
    te["breakdown_uplift"] = p_break_old - p_break_new

    # Aggregate uplift per vehicle for rollout planning
    by_vehicle = te.groupby("vehicle_id").agg(
        mean_fuel_uplift=("fuel_uplift", "mean"),
        mean_breakdown_uplift=("breakdown_uplift", "mean"),
        n_trips=("vehicle_id", "size")
    ).reset_index().sort_values(["mean_fuel_uplift", "mean_breakdown_uplift"], ascending=False)
    by_vehicle["rank"] = np.arange(1, len(by_vehicle)+1)

    # Decile diagnostics for both uplifts (observed effect in each decile)
    def deciles(observed_df, uplift_col, outcome_col, is_binary):
        df_ = observed_df.sort_values(uplift_col, ascending=False).copy()
        df_["uplift_decile"] = pd.qcut(df_[uplift_col], 10, labels=[f"D{i}" for i in range(10,0,-1)])
        out = []
        for dec, g in df_.groupby("uplift_decile"):
            # observed effect = mean(Y|old) - mean(Y|new)
            if is_binary:
                m_old = g.loc[g["T"]==0, outcome_col].mean() if (g["T"]==0).any() else np.nan
                m_new = g.loc[g["T"]==1, outcome_col].mean() if (g["T"]==1).any() else np.nan
                eff = m_old - m_new
            else:
                m_old = g.loc[g["T"]==0, outcome_col].mean()
                m_new = g.loc[g["T"]==1, outcome_col].mean()
                eff = m_old - m_new
            out.append({"uplift_decile": str(dec), "obs_effect": eff,
                        "n_bin": len(g), "n_new": int((g['T']==1).sum()), "n_old": int((g['T']==0).sum())})
        return pd.DataFrame(out).sort_values("uplift_decile", ascending=False)

    fuel_deciles = deciles(te, "fuel_uplift", "Y_fuel", is_binary=False)
    break_deciles = deciles(te, "breakdown_uplift", "Y_break", is_binary=True)

    # Meta-model: what features modulate uplift? (heuristic)
    # Train an RF regressor to predict 'fuel_uplift' from X; extract feature importances.
    meta = te[cats+nums].copy()
    meta["fuel_uplift"] = te["fuel_uplift"].values
    mod = Pipeline(steps=[
        ("prep", pre),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1))
    ])
    mod.fit(meta[cats+nums], meta["fuel_uplift"])
    # Pull feature names from ColumnTransformer
    ohe = mod.named_steps["prep"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cats))
    feature_names = cat_names + nums
    importances = mod.named_steps["rf"].feature_importances_
    feat_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

    # Model details for logging
    details = pd.DataFrame([{
        "fuel_mae_old": mae_old, "fuel_mae_new": mae_new,
        "fuel_r2_old": r2_old, "fuel_r2_new": r2_new,
        "break_auc_old": auc_old, "break_auc_new": auc_new,
        "break_logloss_old": ll_old, "break_logloss_new": ll_new,
        "algo": "T-Learner (RF Regressor on fuel, RF Classifier on breakdown)"
    }])

    return by_vehicle, fuel_deciles, break_deciles, feat_df, details

# =============================
# Visualization
# =============================
def make_plots(df: pd.DataFrame, fuel_deciles: pd.DataFrame, break_deciles: pd.DataFrame):
    # Fuel rate by device
    tmp = df[["new_device", "fuel_rate_l_per_100km"]].dropna().copy()
    tmp["Group"] = np.where(tmp["new_device"], "New Device", "Old Device")
    plt.figure(figsize=(8,5))
    sns.boxplot(data=tmp, x="Group", y="fuel_rate_l_per_100km")
    plt.title("Fuel Rate (L/100km) by Device Generation")
    plt.xlabel("")
    plt.ylabel("Fuel Rate (L/100km)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fuel_rate_by_device.png"))
    plt.close()

    # Breakdown rate by device
    rate = df.groupby("new_device")["breakdown_event"].mean().reset_index()
    rate["Group"] = np.where(rate["new_device"], "New Device", "Old Device")
    plt.figure(figsize=(7,5))
    sns.barplot(data=rate, x="Group", y="breakdown_event")
    for i, r in rate.iterrows():
        plt.text(i, r["breakdown_event"]+0.002, f"{r['breakdown_event']:.3f}", ha="center")
    plt.title("Breakdown Event Rate by Device Generation")
    plt.xlabel("")
    plt.ylabel("Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "breakdown_rate_by_device.png"))
    plt.close()

    # Downtime by device
    tmp2 = df[["new_device", "downtime_hours"]].dropna().copy()
    tmp2["Group"] = np.where(tmp2["new_device"], "New Device", "Old Device")
    plt.figure(figsize=(8,5))
    sns.violinplot(data=tmp2, x="Group", y="downtime_hours", cut=0)
    plt.title("Downtime (Hours) by Device Generation")
    plt.xlabel("")
    plt.ylabel("Downtime (hours)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "downtime_by_device.png"))
    plt.close()

    # Uplift deciles plots
    plt.figure(figsize=(8,5))
    sns.lineplot(data=fuel_deciles, x="uplift_decile", y="obs_effect", marker="o", sort=False)
    plt.title("Observed Fuel-Rate Reduction by Predicted Uplift Decile")
    plt.xlabel("Predicted Uplift Decile (highest uplift on left)")
    plt.ylabel("Observed Reduction in L/100km (Old - New)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "uplift_deciles_fuel_rate.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    sns.lineplot(data=break_deciles, x="uplift_decile", y="obs_effect", marker="o", sort=False)
    plt.title("Observed Breakdown Reduction by Predicted Uplift Decile")
    plt.xlabel("Predicted Uplift Decile (highest uplift on left)")
    plt.ylabel("Observed Reduction in Breakdown Probability (Old - New)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "uplift_deciles_breakdown.png"))
    plt.close()

# =============================
# MAIN
# =============================
def main():
    print("[INFO] Loading fleet_data_cleaned...")
    df = load_data("fleet_data_cleaned")
    print(f"[INFO] Rows loaded: {len(df):,}")

    # Prepare flags and types
    df = add_device_group(df)
    for c in ["breakdown_event", "collision_alert", "harsh_acceleration_event", "braking_event",
              "sensor_fault_event", "gps_loss_event", "network_delay_event"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Drop obvious NaNs in key outcomes used repeatedly
    df = df.dropna(subset=["fuel_rate_l_per_100km", "downtime_hours", "maintenance_cost_usd",
                           "breakdown_event", "device_generation"])

    # --- A/B & Hypothesis testing
    print("[INFO] Running A/B + hypothesis tests...")
    ab = device_ab_tests(df)
    print(ab)
    df_to_files(ab, "dev_ab_summary")
    df_to_postgres(ab, "device_ab_summary")

    # --- Propensity Score Weighting
    print("[INFO] Estimating ATE via stabilized IPW...")
    ipw = ipw_new_device(df, seed=42, bootstrap_iters=200)
    print(ipw)
    df_to_files(ipw, "dev_ipw_summary")
    df_to_postgres(ipw, "device_ipw_summary")

    # --- T-Learner Adoption Effect
    print("[INFO] T-Learner for per-vehicle adoption effect (fuel & breakdown)...")
    by_vehicle, fuel_deciles, break_deciles, feat_modulators, details = t_learner_adoption_effect(df, seed=42)
    df_to_files(by_vehicle, "dev_uplift_by_vehicle")
    df_to_files(fuel_deciles, "dev_uplift_deciles_fuel")
    df_to_files(break_deciles, "dev_uplift_deciles_breakdown")
    df_to_files(feat_modulators, "dev_uplift_feature_modulators")
    df_to_files(details, "dev_model_details")

    df_to_postgres(by_vehicle, "device_uplift_by_vehicle")
    df_to_postgres(fuel_deciles, "device_uplift_deciles_fuel")
    df_to_postgres(break_deciles, "device_uplift_deciles_breakdown")
    df_to_postgres(feat_modulators, "device_uplift_feature_modulators")
    df_to_postgres(details, "device_model_details")

    # --- Plots
    print("[INFO] Creating plots...")
    make_plots(df, fuel_deciles, break_deciles)

    # --- Narrative / Notes (CLI printout)
    print("\n[INTERPRETATION NOTES]")
    try:
        fuel_row = ab[ab["metric"]=="fuel_rate_l_per_100km_mean"].iloc[0]
        print(f"- Fuel rate (L/100km): New={fuel_row['treatment_mean']:.2f}, Old={fuel_row['control_mean']:.2f}, "
              f"Δ(New-Old)={fuel_row['diff_t_minus_c']:.2f}, p={fuel_row['p_value']:.3g} (Welch).")
    except Exception:
        pass
    try:
        brk_row = ab[ab["metric"]=="breakdown_event_rate"].iloc[0]
        print(f"- Breakdown rate: New={brk_row['treatment_mean']:.3f}, Old={brk_row['control_mean']:.3f}, "
              f"Δ={brk_row['diff_t_minus_c']:.3f}, p={brk_row['p_value']:.3g} (chi-square).")
    except Exception:
        pass
    try:
        ipw_fuel = ipw[ipw["outcome"]=="fuel_rate_l_per_100km"].iloc[0]
        print(f"- IPW ATE on fuel rate: Δ(New-Old)={ipw_fuel['diff_t_minus_c']:.3f} "
              f"[{ipw_fuel['ci95_low']:.3f}, {ipw_fuel['ci95_high']:.3f}] (stabilized IPW).")
    except Exception:
        pass

    print("- Uplift tables rank vehicles by *expected* benefit from upgrading: "
          "'dev_uplift_by_vehicle' (fuel & breakdown).")
    print("- 'dev_uplift_feature_modulators' shows which features correlate with larger gains (heuristic).")
    print(f"\n[INFO] Artifacts written to: {OUTPUT_DIR}")
    print("[INFO] ✅ Device comparison analysis complete")

if __name__ == "__main__":
    main()
