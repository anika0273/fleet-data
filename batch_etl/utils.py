# Helper functions for Spark and PostgreSQL connection
# This module provides utility functions to create a Spark session and connect to a PostgreSQL database.    

# batch_etl/utils.py
from pyspark.sql import SparkSession
import os

def get_spark_session(app_name="FleetBatchETL"):
    """
    Returns a Spark Session configured with PostgreSQL JDBC driver.
    """
    postgres_jar_path = os.path.join(os.getcwd(), "lib", "postgresql-42.7.3.jar")
    spark = (SparkSession.builder
             .appName(app_name)
             .config("spark.jars", postgres_jar_path)
             .getOrCreate())
    return spark

def get_jdbc_properties():
    """
    PostgreSQL JDBC connection properties.
    """
    return {
        "url": "jdbc:postgresql://localhost:5433/fleet_db",
        "properties": {
            "user": "postgres",
            "password": "1234",
            "driver": "org.postgresql.Driver"
        }
    }
