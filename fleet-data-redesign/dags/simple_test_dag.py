# dags/simple_test_dag.py

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

DEFAULT_ARGS = {
    "owner": "test_user",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

def print_hello():
    print("Hello from Airflow!")

with DAG(
    dag_id="simple_test_dag",
    default_args=DEFAULT_ARGS,
    description="A simple test DAG to check Airflow job runs",
    schedule_interval="@daily",
    start_date=datetime(2025, 9, 1),
    catchup=False,
    tags=["test"],
) as dag:

    task_hello = PythonOperator(
        task_id="hello_task",
        python_callable=print_hello,
    )
    
    task_bash = BashOperator(
        task_id="bash_task",
        bash_command="echo 'This is a bash task running!'"
    )

    task_hello >> task_bash
