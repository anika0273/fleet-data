# etl/batch/utils.py
# Minimal, production-friendly helpers shared by batch jobs.

import os
from pyspark.sql import SparkSession

# --- Paths ---
# Resolve the project root no matter where this module is called from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
POSTGRES_JAR = os.path.join(PROJECT_ROOT, "lib", "postgresql-42.7.3.jar")

# --- DB config (defaults match your docker-compose.yml mapping) ---
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = int(os.getenv("PGPORT", "5434"))          # <— matches docker-compose.yml
DB_NAME = os.getenv("PGDATABASE", "fleet_db")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD", "1234")

def get_spark_session(app_name: str = "FleetBatchETL") -> SparkSession:
    """
    Return a SparkSession with the PostgreSQL JDBC jar pre-loaded.
    The POSTGRES_JAR path resolves relative to the project root so it
    works from any working directory and in CI.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars", POSTGRES_JAR)
        .getOrCreate()
    )
    return spark

def get_jdbc_properties():
    """
    Return JDBC URL + properties for Spark <-> PostgreSQL writes/reads.
    Values can be overridden via env vars:
      PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD
    """
    return {
        "url": f"jdbc:postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}",
        "properties": {
            "user": DB_USER,
            "password": DB_PASSWORD,
            "driver": "org.postgresql.Driver",
        },
    }
