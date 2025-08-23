# Fleet Data Pipeline – End-to-End Monitoring

This project simulates and monitors a **Fleet Management Data Platform** using modern data engineering and monitoring tools.  
It integrates **PostgreSQL, Kafka, Spark, Prometheus, Grafana** into a single reproducible setup.

---

## 📌 Architecture Overview

1. **Kafka + Zookeeper** – Streams fleet telematics events (e.g., speed, fuel, location).
2. **PostgreSQL** – Stores curated fleet data.
3. **Spark Structured Streaming** – Consumes Kafka events, performs ETL, writes to PostgreSQL.
4. **Prometheus** – Scrapes metrics from exporters (Postgres, Kafka, system, pipeline jobs).
5. **Grafana** – Visualizes both **business KPIs** and **pipeline health metrics**.

---

## ⚡ Components Breakdown

### 🗄️ PostgreSQL (Database)

- Runs inside Docker.
- Accessible on `localhost:5434`.
- Database: `fleet_db`.
- Default user: `postgres / 1234`.

### 🔥 Spark (Processing Engine)

- Runs in **master mode** inside Docker.
- UI available at [http://localhost:8080](http://localhost:8080).
- Executes:
  - Batch ETL (`batch_etl.py`)
  - Streaming ETL (`streaming_etl.py`)

### 📡 Kafka + Zookeeper (Message Broker)

- Zookeeper: `localhost:2181`
- Kafka Broker: `localhost:9092`
- Topic used: `fleet-data`
- Used for real-time ingestion of telematics events.

### 📊 Prometheus (Metrics Collection)

- Runs on [http://localhost:9090](http://localhost:9090).
- Collects metrics from:
  - Node Exporter (CPU, memory, host metrics).
  - PostgreSQL Exporter (query times, DB performance).
  - Kafka Exporter (consumer lag).
  - Custom metrics exposed by ETL jobs.

### 📈 Grafana (Visualization & Dashboards)

- Runs on [http://localhost:3000](http://localhost:3000).
- Default login: `admin / admin`.
- Dashboards:
  - **System Health** (CPU, memory, Kafka lag, Postgres load).
  - **Business KPIs** (driver behavior, route optimization, fuel efficiency).
  - **Pipeline Metrics** (rows processed, streaming latency, error counts).

---

## 🛠️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/fleet-data.git
cd fleet-data
```

### 2. Start Services

Start Core DB + Spark

```bash
docker-compose up -d
```

Start Kafka + Zookeeper

```bash
docker-compose -f docker-compose.kafka.yml up -d
```

Start Monitoring Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Create Kafka Topic

```bash
docker exec -it $(docker ps -qf "name=kafka") kafka-topics \
  --create \
  --topic fleet-data \
  --bootstrap-server localhost:9092 \
  --partitions 1 \
  --replication-factor 1
```

Verify:

```bash
docker exec -it $(docker ps -qf "name=kafka") kafka-topics \
  --list \
  --bootstrap-server localhost:9092
```

Quick sanity check

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 4. Run Producer

```bash
python streaming/producer.py
```

### 5. Run Spark Streaming Job

```bash
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0-preview2 \
  --jars lib/postgresql-42.7.3.jar \
  streaming/streaming_etl.py
```

### 6. Verify Database

```bash
docker exec -it $(docker ps -qf "name=postgres") psql -U postgres -d fleet_db -c "SELECT * FROM fleet_data LIMIT 10;"
```

If you want to drop the table

```bash
PGPASSWORD=1234 psql -h localhost -p 5433 -U postgres -d fleet_db \
  -c "DROP TABLE IF EXISTS fleet_stream_processed;"
```

### 7. Verify Monitoring

- Prometheus → http://localhost:9090
- Grafana → http://localhost:3000

Log in (admin/admin) and add dashboards:

- PostgreSQL metrics (via Postgres Exporter).
- Kafka lag (via Kafka Exporter).
- System metrics (via Node Exporter).
- ETL job metrics (via Prometheus batch_etl and streaming_etl jobs).

## Health Checklist

- docker ps → all services should be Up.
- psql → data flowing into fleet_data table.
- Prometheus Targets → visit http://localhost:9090/targets
  , all jobs should show as UP.
- Grafana dashboards should show live data.

## Troubleshooting

1. Postgres Exporter not connecting? -> Ensure port in docker-compose.monitoring.yml matches 5434.
2. Spark cannot write to Postgres? -> Check driver JAR (postgresql-42.7.3.jar) is mounted correctly.
3. Prometheus shows DOWN target? -> Verify container ports and update prometheus.yml.
