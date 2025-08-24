from dotenv import load_dotenv
import os

# Load .env file if present
load_dotenv()


# project-root relative paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

POSTGRES_JAR = os.path.join(PROJECT_ROOT, "lib", "postgresql-42.7.3.jar")

# Database configuration (can be overridden via environment variables)
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'fleet_db')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '1234')

KAFKA_BROKERS = os.getenv('KAFKA_BROKERS', 'kafka:9092')
SPARK_JDBC_JAR = os.getenv('SPARK_JDBC_JAR', 'lib/postgresql-42.7.3.jar')
