# Generate new data daily + clean old data older than 3 days
# dags/data_generation_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
import psycopg2

from data_generation.generate_data import main as generate_data_main

default_args = {
    'owner': 'fleet_team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def clean_old_data():
    """Delete records older than 3 days to keep DB size manageable."""
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM fleet_data 
        WHERE timestamp < NOW() - INTERVAL '3 days';
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Old data older than 3 days deleted from fleet_data.")

with DAG(
    dag_id='daily_data_generation_with_cleanup',
    default_args=default_args,
    description='Generate fleet data daily and clean old records',
    schedule_interval='@daily',
    start_date=datetime(2025, 8, 26),
    catchup=False,
    tags=['data_generation'],
) as dag:

    generate_data_task = PythonOperator(
        task_id='generate_fleet_data',
        python_callable=generate_data_main
    )

    clean_data_task = PythonOperator(
        task_id='clean_old_data',
        python_callable=clean_old_data
    )

    generate_data_task >> clean_data_task

# Note: To trigger this DAG manually from the command line:
# docker compose up -d airflow_scheduler airflow_webserver
# docker compose run --rm airflow_cli airflow dags trigger daily_data_generation_with_cleanup