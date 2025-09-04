# dags/daily_batch_etl.py
"""
DAG daily_batch_etl.py has two tasks:
1. run_batch_etl → calls your Spark job (batch_etl.py) via spark-submit.
2. data_quality_checks → runs SQL checks on the cleaned table in PostgreSQL.

Notes: ETL first then DQ checks is a common pattern.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from subprocess import run

# Import your ETL entrypoint
# Important: path must be on PYTHONPATH via docker-compose volume mount
# from etl.batch.batch_etl import main as batch_etl_main

DEFAULT_ARGS = {
    "owner": "fleet_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ------------------------
# Spark ETL
# ------------------------
def run_spark_etl(**context):
    """
    Call spark-submit on your batch ETL script and log output.
    """
    result = run(
        [
            "/opt/spark/bin/spark-submit",
            "--master", "spark://spark-master:7077",
            "/opt/airflow/etl/batch/batch_etl.py",
        ],
        capture_output=True,  # capture stdout and stderr
        text=True             # decode bytes → string
    )

    print("----- Spark STDOUT -----")
    print(result.stdout)
    print("----- Spark STDERR -----")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Spark job failed → exit code {result.returncode}")


# ------------------------
# Data Quality Checks
# ------------------------
def quality_checks(**context):
    hook = PostgresHook(postgres_conn_id="postgres_fleet")

    # Ensure rows landed
    row_cnt = hook.get_first("SELECT COUNT(*) FROM fleet_data_cleaned;")[0]
    if row_cnt < 100:
        raise ValueError(f"Row count too low: {row_cnt}")

    # No negative values
    bad_vals = hook.get_first("""
        SELECT COUNT(*) FROM fleet_data_cleaned
        WHERE distance_traveled_km < 0
           OR fuel_consumption_liters < 0
           OR speed < 0
    """)[0]
    if bad_vals > 0:
        raise ValueError(f"Found {bad_vals} rows with invalid negative values.")

    # Required columns not NULL
    nulls = hook.get_first("""
        SELECT COUNT(*) FROM fleet_data_cleaned
        WHERE vehicle_id IS NULL OR timestamp IS NULL
    """)[0]
    if nulls > 0:
        raise ValueError(f"Found {nulls} rows missing critical keys.")

    # Freshness check
    today_rows = hook.get_first("""
        SELECT COUNT(*) FROM fleet_data_cleaned
        WHERE DATE(timestamp) = CURRENT_DATE
    """)[0]
    if today_rows == 0:
        raise ValueError("No rows landed for today → possible upstream issue.")

    print(f"✅ Quality checks passed: {row_cnt} rows, {bad_vals} bad, {nulls} nulls.")

# ------------------------
# DAG Definition
# ------------------------
with DAG(
    dag_id="daily_batch_etl",
    default_args=DEFAULT_ARGS,
    description="Run Spark batch ETL and validate cleaned fleet data",
    schedule_interval="@daily",
    start_date=datetime(2025, 8, 26),
    catchup=False,
    tags=["etl", "spark", "fleet"],
) as dag:

    etl = PythonOperator(
        task_id="run_batch_etl",
        python_callable=run_spark_etl,
        provide_context=True,
    )

    dq = PythonOperator(
        task_id="data_quality_checks",
        python_callable=quality_checks,
        provide_context=True,
    )

    etl >> dq




