import os

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres') # use service name, not localhost
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'fleet_db')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '1234')

KAFKA_BROKERS = os.getenv('KAFKA_BROKERS', 'kafka:9092')
SPARK_JDBC_JAR = os.getenv('SPARK_JDBC_JAR', 'lib/postgresql-42.7.3.jar')
