"""
After building the Docker images, run the following commands to start Airflow and set up the admin user and Postgres connection.

docker compose down -v  # Stop and remove containers, networks, volumes, and images created by up
docker compose up airflow-init
docker compose up -d airflow-webserver airflow-scheduler

# Set the username and Password from CLI:
  docker exec -it fleet-data-airflow-webserver airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

# Login at http://localhost:8080 with username: admin, password: admin

# Create DB connection in Airflow UI:
docker exec -it fleet-data-airflow-webserver airflow connections add fleet_postgres \
  --conn-type postgres \
  --conn-host postgres \
  --conn-schema fleet_db \
  --conn-login postgres \
  --conn-password 1234 \
  --conn-port 5432

# Check connections: docker exec -it fleet-data-airflow-webserver airflow db check
"""
# dags/generate_data_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import logging

logger = logging.getLogger("airflow.task")
logger.info(f"sys.path at DAG parse time: {sys.path}")

# We add '/opt/airflow' to sys.path to tell Python 
# where to look for our custom modules (like config and data_generation)
# because by default, when Airflow loads DAGs inside Docker,
# it may not include our project folder in the search path.
#
# This ensures Python can find and import our local packages correctly.

# Add root airflow folder to path before anything else
sys.path.insert(0, '/opt/airflow')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
from data_generation.generate_data import main as generate_data_main

def log_sys_path():
    logger.info(f"sys.path at task runtime: {sys.path}")

default_args = {
    "owner": "data_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="generate_synthetic_data",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 9, 5),
    catchup=False,
) as dag:

    # Task to generate synthetic data
    generate_task = PythonOperator(
        task_id="generate_1000_data_rows",
        python_callable=generate_data_main,
    )

    # Task to log sys.path during task execution for debugging
    log_path_task = PythonOperator(
        task_id="log_sys_path_task",
        python_callable=log_sys_path,
    )

    # Define task dependencies if needed (optional)
    log_path_task >> generate_task
