# dags/daily_batch_etl.py
"""
Airflow DAG: Daily Batch ETL
Runs your Spark-based batch ETL (etl/batch/batch_etl.py) and validates results.

Flow:
1. Run Spark ETL (Extract → Transform → Load).
2. Run quality checks on fleet_data_cleaned table in Postgres.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Import your ETL entrypoint
# Important: path must be on PYTHONPATH via docker-compose volume mount
from etl.batch.batch_etl import main as batch_etl_main

DEFAULT_ARGS = {
    "owner": "fleet_team",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ------------------------
# Tasks
# ------------------------
def run_batch_etl(**context):
    """
    Kick off Spark ETL job (batch_etl.main()).
    """
    # Could pass execution_date if you partition parquet by date
    batch_etl_main()


def quality_checks(**context):
    """
    Post-load validation on fleet_data_cleaned in Postgres.
    Ensures job didn’t silently fail and data is usable.
    """
    hook = PostgresHook(postgres_conn_id="postgres_fleet")

    # 1) Ensure rows landed
    row_cnt = hook.get_first("SELECT COUNT(*) FROM fleet_data_cleaned;")[0]
    if row_cnt < 100:
        raise ValueError(f"Row count too low: {row_cnt}")

    # 2) No negative values
    bad_vals = hook.get_first("""
        SELECT COUNT(*) FROM fleet_data_cleaned
        WHERE distance_traveled_km < 0
           OR fuel_consumption_liters < 0
           OR speed < 0
    """)[0]
    if bad_vals > 0:
        raise ValueError(f"Found {bad_vals} rows with invalid negative values.")

    # 3) Required columns are not NULL
    nulls = hook.get_first("""
        SELECT COUNT(*) FROM fleet_data_cleaned
        WHERE vehicle_id IS NULL OR timestamp IS NULL
    """)[0]
    if nulls > 0:
        raise ValueError(f"Found {nulls} rows missing critical keys.")

    # 4) Freshness check — did we load data today?
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
        python_callable=run_batch_etl,
        provide_context=True,
    )

    dq = PythonOperator(
        task_id="data_quality_checks",
        python_callable=quality_checks,
        provide_context=True,
    )

    etl >> dq


# run `docker compose up --build -d` to start all services including Airflow
# Access Airflow UI at http://localhost:8080 (admin/admin)  
# To initialize the Airflow database, run:
# docker-compose run --rm airflow-init
# Start the webserver and scheduler:
# docker compose up -d airflow-webserver airflow-scheduler  
# To stop all services, run:
# docker compose down
