# etl/batch/utils.py
# Minimal, production-friendly helpers shared by batch jobs.

import os
from pyspark.sql import SparkSession

from config.config import (
    POSTGRES_JAR, POSTGRES_HOST, POSTGRES_PORT,
    POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
)

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
        "url": f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        "properties": {
            "user": POSTGRES_USER,
            "password": POSTGRES_PASSWORD,
            "driver": "org.postgresql.Driver",
        },
    }
