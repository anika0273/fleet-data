# Fleet Data Project - README

## Environment Setup So Far

Created a new project folder with the following structure:

```text
fleet-data/
├── config/
│   └── config.py                # Centralized environment configs
├── etl/
├── analysis/
├── data_generation/
├── monitoring/
│   ├── prometheus.yml           # Prometheus config
│   ├── grafana/
│   │   └── provisioning/        # Grafana dashboards, datasources
├── lib/                        # JDBC driver
├── docker-compose.yml           # Single Docker Compose file combining all services
├── requirements.txt
└── .gitignore
```

### docker-compose.yml includes services for:

1. Postgres (database)
2. Spark (ETL compute)
3. Kafka + Zookeeper (streaming)
4. Prometheus + Grafana (monitoring and visualization)
5. Exporters for Prometheus to scrape Postgres, Kafka metrics

Prometheus and Grafana configs are mounted from monitoring/ folder.

Centralized configurations are defined in config/config.py which all code imports for connection parameters.

## How I Checked Everything Is Working

Launched all infrastructure containers with:

```bash
docker compose up -d
```

Verified all containers are running:

```bash
docker compose ps
```

Connected to Docker Postgres:

```bash
psql -h localhost -p 5434 -U postgres -d fleet_db
```

Initially empty since no tables have been created yet.

Opened web UI for:

- Spark: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default user: admin, password: admin)

Ensured Python scripts load configuration properly from config.config.

## Initial Files Needed at Project Start

1. config/config.py — environment variables like DB host, port, user, password.
2. docker-compose.yml — to start all services in a single command.
3. monitoring/prometheus.yml & monitoring/grafana/provisioning/ — for monitoring and dashboards.
4. lib/postgresql-42.7.3.jar — PostgreSQL JDBC driver for Spark.
5. data_generation/generate_data.py — synthetic data generator.
6. etl/batch/batch_etl.py and etl/batch/utils.py — batch ETL scripts.
7. .gitignore and requirements.txt.

## Next Steps

1. Run synthetic data generator to populate Postgres with initial fleet data.

2. Run batch ETL to clean and transform data.

3. Proceed with streaming ETL and business analytics modules.

4. Monitor system health via Grafana dashboards.
