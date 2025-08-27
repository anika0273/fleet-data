"""
feedback_system.py

Objective (Story)
-----------------
4) Does providing *real-time feedback* to drivers (intervention_active=True) reduce
   harsh acceleration/braking events and downstream wear (maintenance cost, downtime)?

Business Value
--------------
- Less wear-and-tear → lower maintenance costs and downtime.
- Fewer harsh events → improved safety statistics.
- Evidence to prioritize continued rollout of in-cab feedback or alerts.

Data Source
-----------
- Static table: `fleet_data_cleaned` in PostgreSQL (Spark ETL wrote this).
- Key fields used here:
    * intervention_active (bool) ........ treatment indicator (feedback delivered this trip)
    * harsh_acceleration_event (bool) ... primary safety metric (treated should reduce it)
    * braking_event (bool) .............. secondary proxy for driving aggressiveness
    * maintenance_cost_usd (float) ...... wear proxy (should decrease)
    * downtime_hours (float) ............ operations impact (should decrease)
    * Plus context/driver/vehicle/device covariates for adjustment & learning.

Approach (What & Why)
---------------------
A) **A/B comparison (descriptive + hypothesis tests)**
   Compare event rates and wear metrics between intervention vs control.
   - Chi-square test for differences in proportions (harsh_acceleration_event, braking_event).
   - Welch t-test & Mann–Whitney for maintenance_cost_usd and downtime_hours.
   This answers: "Is there a statistically significant reduction?"

B) **Causal adjustment via Propensity Score Weighting (ATE/ATT)**
   Real deployments often have non-random uptake (e.g., certain drivers or routes
   might use feedback more often). To reduce bias, we:
   - Fit a logistic regression to estimate P(Treatment | X) using reasonable covariates.
   - Compute stabilized inverse probability weights and estimate ATT/ATE on
     harsh events and cost metrics.
   This answers: "Is the reduction robust after controlling for confounders?"

C) **Uplift Modeling (Who benefits most?)**
   Product goal: target drivers/contexts where feedback yields the *largest* reduction
   in harsh events. We use a simple **T-Learner**:
   - Train two models: P(Y=1 | X, T=0) and P(Y=1 | X, T=1).
   - Predict uplift = p0 - p1 (expected reduction under treatment).
   - Summarize uplift by driver_id and produce decile diagnostics.
   This answers: "Where should we focus/scale the feedback to get maximum ROI?"

Outputs
-------
- All outputs saved to: fleet-data/analysis_outputs/feedback_system/
    * CSV + Parquet tables:
        - fb_ab_summary.csv/parquet .............. A/B test results (proportions & means)
        - fb_ps_weighting_summary.csv/parquet .... Weighted ATE/ATT estimates
        - fb_uplift_by_driver.csv/parquet ........ Avg predicted uplift by driver (target list)
        - fb_uplift_deciles.csv/parquet .......... Observed reduction by uplift decile (diagnostic)
        - fb_model_details.csv/parquet ........... Model metadata & quick metrics
    * PostgreSQL tables (if_exists="replace"):
        - feedback_ab_summary
        - feedback_ps_weighting_summary
        - feedback_uplift_by_driver
        - feedback_uplift_deciles
        - feedback_model_details
    * Plots (PNG):
        - harsh_rate_by_intervention.png
        - maint_cost_by_intervention.png
        - uplift_deciles_observed_reduction.png
        - treatment_propensity_hist.png

How to Run
----------
$ python feedback_system.py
(Ensure Postgres is up and `fleet_data_cleaned` exists)

"""

import os, sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

# ==================================
# Config: PostgreSQL + Output paths
# ==================================

from config import config

DB_CONFIG = {
    "host": config.POSTGRES_HOST,
    "port": config.POSTGRES_PORT,
    "database": config.POSTGRES_DB,
    "user": config.POSTGRES_USER,
    "password": config.POSTGRES_PASSWORD
}

# analysis_outputs at repo root (sibling to analysis/)
OUTPUT_DIR = os.path.join(config.PROJECT_ROOT, "outputs", "feedback_system")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =============================

def get_connection():
    """Create a SQLAlchemy engine using your existing DSN style."""
    url = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@" \
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    return create_engine(url)

# =============================
# Load Data
# =============================
def load_data(table: str = "fleet_data_cleaned") -> pd.DataFrame:
    """Load cleaned fleet dataset from PostgreSQL."""
    engine = get_connection()
    return pd.read_sql(f"SELECT * FROM {table};", engine)

# =============================
# Utilities (IO)
# =============================
def df_to_files(df: pd.DataFrame, name: str):
    """Save a DataFrame to CSV and Parquet in OUTPUT_DIR."""
    df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
    df.to_parquet(os.path.join(OUTPUT_DIR, f"{name}.parquet"), index=False)

def df_to_postgres(df: pd.DataFrame, table_name: str, if_exists="replace"):
    """Write a DataFrame to PostgreSQL (replace by default)."""
    eng = get_connection()
    df.to_sql(table_name, eng, index=False, if_exists=if_exists)

# =============================
# A/B Testing (Descriptive + HT)
# =============================
def ab_testing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare intervention (treatment) vs control on:
        - harsh_acceleration_event (rate)  [primary]
        - braking_event (rate)             [secondary]
        - maintenance_cost_usd (mean/median)
        - downtime_hours (mean/median)

    Tests:
        - Chi-square for proportions (harsh, braking)
        - Welch t-test + Mann-Whitney for costs/downtime
    """
    # Ensure types
    for c in ["harsh_acceleration_event", "braking_event", "intervention_active"]:
        df[c] = df[c].astype(bool)

    # Groups
    treat = df[df["intervention_active"]]
    ctrl  = df[~df["intervention_active"]]

    rows = []

    # --- Proportions: harsh_acceleration_event
    tab = pd.crosstab(df["intervention_active"], df["harsh_acceleration_event"])
    if tab.shape == (2,2):
        chi2, p_chi, _, _ = stats.chi2_contingency(tab.values)
    else:
        chi2, p_chi = np.nan, np.nan
    rate_t = treat["harsh_acceleration_event"].mean()
    rate_c = ctrl["harsh_acceleration_event"].mean()
    rows.append({
        "metric": "harsh_acceleration_event_rate",
        "treatment_mean": rate_t,
        "control_mean": rate_c,
        "diff_t_minus_c": rate_t - rate_c,
        "test": "chi2",
        "stat": chi2,
        "p_value": p_chi,
        "n_treat": len(treat), "n_control": len(ctrl)
    })

    # --- Proportions: braking_event (secondary)
    tab = pd.crosstab(df["intervention_active"], df["braking_event"])
    if tab.shape == (2,2):
        chi2, p_chi, _, _ = stats.chi2_contingency(tab.values)
    else:
        chi2, p_chi = np.nan, np.nan
    rate_t = treat["braking_event"].mean()
    rate_c = ctrl["braking_event"].mean()
    rows.append({
        "metric": "braking_event_rate",
        "treatment_mean": rate_t,
        "control_mean": rate_c,
        "diff_t_minus_c": rate_t - rate_c,
        "test": "chi2",
        "stat": chi2,
        "p_value": p_chi,
        "n_treat": len(treat), "n_control": len(ctrl)
    })

    # --- Means: maintenance_cost_usd
    t_stat, p_t = stats.ttest_ind(treat["maintenance_cost_usd"], ctrl["maintenance_cost_usd"], equal_var=False, nan_policy="omit")
    u_stat, p_u = stats.mannwhitneyu(treat["maintenance_cost_usd"], ctrl["maintenance_cost_usd"], alternative="two-sided")
    rows.append({
        "metric": "maintenance_cost_usd_mean",
        "treatment_mean": treat["maintenance_cost_usd"].mean(),
        "control_mean": ctrl["maintenance_cost_usd"].mean(),
        "diff_t_minus_c": treat["maintenance_cost_usd"].mean() - ctrl["maintenance_cost_usd"].mean(),
        "test": "welch_t",
        "stat": t_stat, "p_value": p_t,
        "mw_u": u_stat, "mw_p": p_u,
        "n_treat": len(treat), "n_control": len(ctrl)
    })

    # --- Means: downtime_hours
    t_stat, p_t = stats.ttest_ind(treat["downtime_hours"], ctrl["downtime_hours"], equal_var=False, nan_policy="omit")
    u_stat, p_u = stats.mannwhitneyu(treat["downtime_hours"], ctrl["downtime_hours"], alternative="two-sided")
    rows.append({
        "metric": "downtime_hours_mean",
        "treatment_mean": treat["downtime_hours"].mean(),
        "control_mean": ctrl["downtime_hours"].mean(),
        "diff_t_minus_c": treat["downtime_hours"].mean() - ctrl["downtime_hours"].mean(),
        "test": "welch_t",
        "stat": t_stat, "p_value": p_t,
        "mw_u": u_stat, "mw_p": p_u,
        "n_treat": len(treat), "n_control": len(ctrl)
    })

    return pd.DataFrame(rows)

# =============================
# Propensity Score Weighting
# =============================
def propensity_weighting(df: pd.DataFrame, seed=42, bootstrap_iters=200) -> pd.DataFrame:
    """
    Estimate stabilized IPW weights for intervention assignment:
        p = P(T=1 | X)
        sw = T * pbar/p + (1-T) * (1-pbar)/(1-p)
    Then compute weighted ATE & ATT for target outcomes:
        - harsh_acceleration_event
        - maintenance_cost_usd
        - downtime_hours
    Bootstrap CIs for robustness (optional but useful in practice).
    """
    # Define covariates likely correlated with T and Y
    # (road/traffic/weather/time/driver/vehicle/device)
    covars_cat = ["road_type", "traffic_density", "weather", "vehicle_type", "driver_training"]
    covars_num = ["hour_of_day", "driver_experience_years", "speed", "idle_time_minutes",
                  "vehicle_age_years", "device_generation", "data_latency_ms",
                  "gps_accuracy_meters", "packet_loss_rate", "sensor_battery", "distance_traveled_km",
                  "fuel_rate_l_per_100km", "time_since_last_maintenance_days"]
    keep_cols = ["intervention_active", "harsh_acceleration_event",
                 "maintenance_cost_usd", "downtime_hours"] + covars_cat + covars_num

    d = df[keep_cols].dropna().copy()
    d["T"] = d["intervention_active"].astype(int)

    # Preprocess (one-hot for categorical)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), covars_cat),
            ("num", "passthrough", covars_num)
        ],
        remainder="drop"
    )

    # Logistic regression for propensity (robust & fast)
    prop_model = Pipeline(steps=[
        ("prep", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight=None, n_jobs=None))
    ])
    X = d[covars_cat + covars_num]
    y = d["T"]

    prop_model.fit(X, y)
    p = prop_model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-3, 1-1e-3)     # avoid extreme weights
    pbar = y.mean()
    d["sw"] = np.where(d["T"]==1, pbar/p, (1-pbar)/(1-p))

    # Helper to compute weighted means and diff
    def wt_diff(y_col, dframe):
        m_t = np.average(dframe.loc[dframe["T"]==1, y_col], weights=dframe.loc[dframe["T"]==1, "sw"])
        m_c = np.average(dframe.loc[dframe["T"]==0, y_col], weights=dframe.loc[dframe["T"]==0, "sw"])
        return m_t, m_c, m_t - m_c

    rows = []
    for y_col, y_kind in [
        ("harsh_acceleration_event", "proportion"),
        ("maintenance_cost_usd", "mean"),
        ("downtime_hours", "mean")
    ]:
        m_t, m_c, diff = wt_diff(y_col, d)

        # Bootstrap simple CI on diff
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(bootstrap_iters):
            samp = d.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
            _, _, di = wt_diff(y_col, samp)
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

    # Quick diagnostic: distribution of propensities & weights → saved as plot
    plt.figure(figsize=(7,5))
    sns.histplot(p, bins=30, kde=True)
    plt.title("Treatment Propensity Distribution P(T=1|X)")
    plt.xlabel("Propensity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "treatment_propensity_hist.png"))
    plt.close()

    return pd.DataFrame(rows)

# =============================
# Uplift Modeling (T-Learner)
# =============================
def t_learner_uplift(df: pd.DataFrame, seed=42):
    """
    Two-model approach to estimate individual treatment effect (uplift) on:
        Y := harsh_acceleration_event (1 = harsh event)
    Uplift = p0(X) - p1(X), where:
        p1(X) = P(Y=1 | X, T=1)  treated model
        p0(X) = P(Y=1 | X, T=0)  control model
    Positive uplift means expected *reduction* in harsh events if treated (good).

    We aggregate per driver_id to produce a target list of drivers who should
    get (or keep) the feedback feature more frequently.
    """
    # Features
    cat = ["road_type", "traffic_density", "weather", "vehicle_type", "driver_training"]
    num = ["hour_of_day", "driver_experience_years", "speed", "idle_time_minutes",
           "vehicle_age_years", "device_generation", "data_latency_ms",
           "gps_accuracy_meters", "packet_loss_rate", "sensor_battery",
           "distance_traveled_km", "fuel_rate_l_per_100km",
           "time_since_last_maintenance_days", "rush_hour"]  # rush_hour already 0/1 in cleaned data

    use_cols = ["driver_id", "intervention_active", "harsh_acceleration_event"] + cat + num
    d = df[use_cols].dropna().copy()

    # Binary labels & treatment
    d["Y"] = d["harsh_acceleration_event"].astype(int)
    d["T"] = d["intervention_active"].astype(int)

    # Split to avoid leakage in evaluation of uplift deciles
    train_idx, test_idx = train_test_split(d.index, test_size=0.25, random_state=seed, stratify=d[["T", "Y"]])
    tr = d.loc[train_idx].copy()
    te = d.loc[test_idx].copy()

    # Preprocessors
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )

    # Control model: on T=0
    model_c = Pipeline(steps=[
        ("prep", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_split=4, min_samples_leaf=2,
            random_state=seed, n_jobs=-1, class_weight="balanced_subsample"))
    ])
    tr_c = tr[tr["T"]==0]
    Xc, yc = tr_c[cat+num], tr_c["Y"]
    model_c.fit(Xc, yc)

    # Treated model: on T=1
    model_t = Pipeline(steps=[
        ("prep", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_split=4, min_samples_leaf=2,
            random_state=seed, n_jobs=-1, class_weight="balanced_subsample"))
    ])
    tr_t = tr[tr["T"]==1]
    Xt, yt = tr_t[cat+num], tr_t["Y"]
    model_t.fit(Xt, yt)

    # Evaluate classification sanity (not uplift metric): AUC/logloss on each segment
    auc_c = roc_auc_score(te[te["T"]==0]["Y"], model_c.predict_proba(te[te["T"]==0][cat+num])[:,1]) if (te["T"]==0).any() else np.nan
    auc_t = roc_auc_score(te[te["T"]==1]["Y"], model_t.predict_proba(te[te["T"]==1][cat+num])[:,1]) if (te["T"]==1).any() else np.nan
    ll_c  = log_loss(te[te["T"]==0]["Y"], model_c.predict_proba(te[te["T"]==0][cat+num])[:,1], labels=[0,1]) if (te["T"]==0).any() else np.nan
    ll_t  = log_loss(te[te["T"]==1]["Y"], model_t.predict_proba(te[te["T"]==1][cat+num])[:,1], labels=[0,1]) if (te["T"]==1).any() else np.nan

    # Predicted uplift on test
    p1 = model_t.predict_proba(te[cat+num])[:, 1]
    p0 = model_c.predict_proba(te[cat+num])[:, 1]
    uplift = p0 - p1
    te = te.copy()
    te["uplift_pred"] = uplift

    # Aggregate uplift by driver_id (mean per driver)
    uplift_by_driver = (te.groupby("driver_id")["uplift_pred"]
                          .mean()
                          .reset_index()
                          .sort_values("uplift_pred", ascending=False))
    uplift_by_driver["uplift_rank"] = np.arange(1, len(uplift_by_driver)+1)

    # Decile diagnostic: sort by uplift, split into 10 bins, compute *observed* reduction
    te = te.sort_values("uplift_pred", ascending=False).reset_index(drop=True)
    te["uplift_decile"] = pd.qcut(te["uplift_pred"], q=10, labels=[f"D{i}" for i in range(10,0,-1)])
    dec = []
    for decile, g in te.groupby("uplift_decile"):
        # observed reduction = mean(Y|T=0) - mean(Y|T=1) within the bin
        m_c = g.loc[g["T"]==0, "Y"].mean() if (g["T"]==0).any() else np.nan
        m_t = g.loc[g["T"]==1, "Y"].mean() if (g["T"]==1).any() else np.nan
        dec.append({"uplift_decile": str(decile), "obs_reduction": (m_c - m_t),
                    "n_bin": len(g), "n_treat": int((g['T']==1).sum()), "n_ctrl": int((g['T']==0).sum())})
    uplift_deciles = pd.DataFrame(dec).sort_values("uplift_decile", ascending=False)

    # Basic model notes
    model_details = pd.DataFrame([{
        "segment_auc_control": auc_c,
        "segment_auc_treated": auc_t,
        "segment_logloss_control": ll_c,
        "segment_logloss_treated": ll_t,
        "algo": "T-Learner (RandomForest)"
    }])

    return uplift_by_driver, uplift_deciles, model_details

# =============================
# Visualization
# =============================
def make_plots(df: pd.DataFrame, uplift_deciles: pd.DataFrame):
    # 1) Harsh rate by intervention (bar)
    rates = df.groupby("intervention_active")["harsh_acceleration_event"].mean().reset_index()
    rates["Group"] = np.where(rates["intervention_active"], "Feedback ON", "Feedback OFF")
    plt.figure(figsize=(7,5))
    sns.barplot(data=rates, x="Group", y="harsh_acceleration_event")
    for i, r in rates.iterrows():
        plt.text(i, r["harsh_acceleration_event"]+0.002, f"{r['harsh_acceleration_event']:.3f}", ha="center")
    plt.title("Harsh Acceleration Event Rate by Intervention")
    plt.ylabel("Rate")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "harsh_rate_by_intervention.png"))
    plt.close()

    # 2) Maintenance cost by intervention (box)
    plt.figure(figsize=(8,5))
    tmp = df.copy()
    tmp["Group"] = np.where(tmp["intervention_active"], "Feedback ON", "Feedback OFF")
    sns.boxplot(data=tmp, x="Group", y="maintenance_cost_usd")
    plt.title("Maintenance Cost by Intervention")
    plt.ylabel("Maintenance Cost (USD)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "maint_cost_by_intervention.png"))
    plt.close()

    # 3) Uplift deciles observed reduction (line+markers)
    plt.figure(figsize=(8,5))
    order = list(uplift_deciles.sort_values("uplift_decile", ascending=False)["uplift_decile"])
    sns.lineplot(data=uplift_deciles, x="uplift_decile", y="obs_reduction", marker="o", sort=False)
    plt.title("Observed Reduction (Control - Treated) by Predicted Uplift Decile")
    plt.xlabel("Predicted Uplift Decile (Highest uplift on left)")
    plt.ylabel("Observed Reduction in Harsh Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "uplift_deciles_observed_reduction.png"))
    plt.close()

# =============================
# MAIN
# =============================
def main():
    print("[INFO] Loading data from PostgreSQL 'fleet_data_cleaned'...")
    df = load_data("fleet_data_cleaned")

    # Ensure expected dtypes
    bool_cols = ["intervention_active", "harsh_acceleration_event", "braking_event"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Drop obvious NaNs in key fields (keep it light; ETL already cleaned)
    df = df.dropna(subset=["intervention_active", "harsh_acceleration_event",
                           "maintenance_cost_usd", "downtime_hours"])

    print("[INFO] A/B testing: event rates and wear proxies")
    ab = ab_testing_summary(df)
    print(ab)
    df_to_files(ab, "fb_ab_summary")
    df_to_postgres(ab, "feedback_ab_summary")

    print("[INFO] Propensity score weighting (stabilized IPW) for robustness")
    ps = propensity_weighting(df, seed=42, bootstrap_iters=200)
    print(ps)
    df_to_files(ps, "fb_ps_weighting_summary")
    df_to_postgres(ps, "feedback_ps_weighting_summary")

    print("[INFO] Uplift modeling (T-Learner) to find who benefits most")
    uplift_by_driver, uplift_deciles, model_details = t_learner_uplift(df, seed=42)
    df_to_files(uplift_by_driver, "fb_uplift_by_driver")
    df_to_files(uplift_deciles, "fb_uplift_deciles")
    df_to_files(model_details, "fb_model_details")

    df_to_postgres(uplift_by_driver, "feedback_uplift_by_driver")
    df_to_postgres(uplift_deciles, "feedback_uplift_deciles")
    df_to_postgres(model_details, "feedback_model_details")

    print("[INFO] Visualization")
    make_plots(df, uplift_deciles)

    # Narrative Notes (for run logs / notebooks)
    print("\n[INTERPRETATION NOTES]")
    # From AB
    ab_harsh = ab[ab["metric"]=="harsh_acceleration_event_rate"].iloc[0]
    print(f"- Harsh event rate: Feedback ON={ab_harsh['treatment_mean']:.3f}, "
          f"OFF={ab_harsh['control_mean']:.3f}, Δ={ab_harsh['diff_t_minus_c']:.3f}, "
          f"p={ab_harsh['p_value']:.3g} (chi-square).")
    ab_cost = ab[ab["metric"]=="maintenance_cost_usd_mean"].iloc[0]
    print(f"- Maintenance cost: ON={ab_cost['treatment_mean']:.2f}, "
          f"OFF={ab_cost['control_mean']:.2f}, Δ={ab_cost['diff_t_minus_c']:.2f}, "
          f"p={ab_cost['p_value']:.3g} (Welch).")
    # From PS weighting
    ps_harsh = ps[ps["outcome"]=="harsh_acceleration_event"].iloc[0]
    print(f"- IPW ATE on harsh events: Δ(ON-OFF)={ps_harsh['diff_t_minus_c']:.4f} "
          f"[{ps_harsh['ci95_low']:.4f}, {ps_harsh['ci95_high']:.4f}] (stabilized IPW).")
    # From uplift
    print("- Uplift model ranks drivers by expected reduction in harsh events when feedback is ON.")
    print("  Use 'fb_uplift_by_driver' to target the top deciles first (highest ROI).")

    print(f"[INFO] All outputs written to: {OUTPUT_DIR}")
    print("[INFO] ✅ Feedback system analysis complete")

if __name__ == "__main__":
    main()
