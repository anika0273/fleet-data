# Fleet Data Monitoring Stack

This project contains a monitoring stack for the Fleet Data platform, using Docker Compose to orchestrate multiple services including Kafka, Zookeeper, PostgreSQL, Prometheus, and Grafana. It also includes provisioning of Grafana dashboards for real-time data visualization.

## Services Included

1. PostgreSQL (with database fleet_db) for storing fleet data.
2. Zookeeper and Kafka for streaming data collection.
3. Prometheus for metrics scraping and monitoring.
4. Postgres Exporter to expose PostgreSQL metrics to Prometheus.
5. Grafana for dashboard visualization with anonymous public access enabled.
6. Data generator and ETL services for feeding synthetic streaming data into PostgreSQL.

## Prerequisites

- Docker and Docker Compose installed on your system.
- Port 3000, 5434, 9090, and other necessary ports free or properly forwarded.

## How to Run the Stack

1. Clone the Repo:

```bash
git clone https://github.com/yourusername/fleet-data-redesign.git
cd fleet-data-redesign
```

2. Start all services using Docker Compose:

```bash
docker-compose up -d
```

Services will start in the following order and may take a few minutes to fully initialize:

- Zookeeper and Kafka
- PostgreSQL and Postgres Exporter
- Prometheus
- Grafana
- Data Generator and ETL Workers

3. Accessing Grafana Dashboards
   Once everything is running, open your browser and navigate to:

```text
http://localhost:3000
```

You can view dashboards without logging in since anonymous access is enabled.

The dashboards will be automatically loaded via provisioning from:

```text
./monitoring/grafana/provisioning/dashboards/
```

The default admin user (if needed) is:

```
text
username: admin
password: admin
```

## Adding or Updating Dashboards

- Export additional dashboards or make changes in Grafana UI.
- Save the dashboard JSON files into:

```text
./monitoring/grafana/provisioning/dashboards/
```

- Restart the Grafana container to reload dashboards:

```bash
docker-compose restart grafana
```

## Prometheus Configuration

Prometheus configuration live at:

```
text
./monitoring/prometheus.yml
```

It scrapes metrics from:

- Prometheus itself (localhost:9090)
- Postgres Exporter (postgres_exporter:9187)

## Customization and Extending

- Modify Docker Compose or add new services as needed.
- Configure additional exporters or data sources in Prometheus and Grafana.
- For development, Python ETL scripts and synthetic data generators are under:

```
text
./etl/
./data_generation/
```

## Troubleshooting

Check container logs for errors:

```
bash
docker-compose logs -f
```

Check container health status:

```
bash
docker ps
```

Restart problematic containers individually:

```
bash
docker-compose restart <service_name>
```

## License and Contributions

Open-source, contributions welcome!

Please open issues or pull requests for improvements.
