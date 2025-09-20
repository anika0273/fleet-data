# Batch ETL Pipeline for Fleet and Supply Chain Analytics

## Overview

This batch ETL pipeline processes synthetic fleet telemetry and supply chain data to produce clean, enriched, and aggregated datasets suitable for downstream analytics, reporting, and machine learning.

It handles large data volumes by running periodic jobs that extract raw data, apply cleaning and complex transformations, calculate derived features, aggregate key metrics, and then load results back into PostgreSQL and Parquet files.

## Purpose

- Prepare historical data to answer strategic questions such as:
  - Impact of driver behavior interventions.
  - Effectiveness of route optimization.
  - Benefits of predictive maintenance.
  - Evaluating different telematics device generations.
  - Detailed shipment delay and ETA prediction analytics.

## Architecture & Workflow

1. **Extract**:

   - Connect to PostgreSQL database.
   - Load raw records with pre-filters applied (e.g., valid GPS coordinates, realistic speed and battery values).

2. **Transform**:

   - Cast data types for consistency.
   - Clean invalid or outlier data.
   - Normalize categorical fields.
   - Generate derived features like:
     - Hour of day (`hour_of_day`).
     - Vehicle age groups (`vehicle_age_group`).
     - Fuel efficiency (`fuel_efficiency_l_per_km`).
     - Downtime flags (`downtime_flag`).
     - Shipment delay severity and late shipment indicators.
   - Prepare data for analytics and ML consumption.

3. **Aggregate**:

   - Calculate counts of safety events.
   - Compute average metrics like speed, battery, fuel efficiency.
   - Summarize shipment delay statistics for performance monitoring.

4. **Load**:
   - Overwrite a clean table in PostgreSQL (`fleet_data_clean`).
   - Save cleaned data as Parquet files for scalable analytics.

## Key Notes

- Runs as a scheduled or manual batch job.
- Uses Apache Spark for scalability and speed on bulk data.
- Exposes Prometheus metrics for monitoring ETL performance.
- Transformation is comprehensive to support rich business questions and ML feature generation.

## How to Run

- Use the provided Docker Compose service or run locally via:

`bash
docker compose run --rm batch_etl python -m etl.batch.batch_etl
`

---

## Prerequisites

- **Apache Spark** (3.x recommended), with Java 17+ installed (`JAVA_HOME` set)
- **Python 3.8+**
- **pyspark** Python package (`pip install pyspark`)
- **PostgreSQL** running locally (with `fleet_db`, and `fleet_data` loaded)
- `postgresql-42.7.3.jar` in a `lib/` folder (required for Spark to connect via JDBC)
- (Optional) Sufficient RAM/CPU for processing >100k records

---

## How Each Part of the ETL Works

### **1. Extract (read_from_postgres)**

- **What it does:**  
   Reads the raw, synthetic fleet telemetry data from your PostgreSQL database table `fleet_data` using JDBC.

- **Key logic:**
  - Applies “pushdown filtering” (removes invalid/missing values at the DB before transfer)
  - Loads only valid rows (null GPS or `speed > 200` rows are excluded at source)
  - Reduces network and memory load for Spark

---

### **2. Transform (clean_and_transform, aggregate_metrics)**

- **Cleaning steps include:**

  - Makes sure `sensor_battery` is a double for calculations
  - Removes rows if battery is unrealistic (`<0` or `>100`)
  - Normalizes text fields (`weather` to lowercase: e.g. `'SUNNY'` and `'sunny'` are treated the same)
  - Adds `hour_of_day` (if not present) for time-based analysis

- **Aggregation (aggregate_metrics):**
  - Groups by `event_type`
  - Counts events, calculates average speed and battery level for each event type
  - Useful for dashboarding/monitoring data coverage and quality

---

### **3. Load (write_to_postgres, write_to_parquet)**

- **Write cleaned data to PostgreSQL**
  - Table: `fleet_data_cleaned` (cleaned, analytics-friendly version)
  - Uses batch-writing to speed up transfer
- **Save as local Parquet file**
  - File path: `data/output/cleaned_fleet_data.parquet` (creates dirs as needed)
  - Parquet format is highly compressed, efficient for Spark/Pandas/BI tools

---

## Step-by-Step: Running the Pipeline Locally

**1. Prepare folders and files**

- Ensure you have the following structure:
  ```
  batch_etl/
    ├─ batch_etl.py
    ├─ utils.py
    └─ lib/
         └─ postgresql-42.7.3.jar
  ```

**2. Install Python dependencies**

```bash
   pip install pyspark
```

**3. Confirm database is running**

```bash
pg_isready -h localhost -p 5433
```

```text
You should see:
`localhost:5433 - accepting connections`
```

**4. Run the ETL pipeline**

```python
python batch_etl/batch_etl.py
```

**5. After the run completes**

- Check your database for the new `fleet_data_cleaned` table:
  ```
  SELECT COUNT(*) FROM fleet_data_cleaned;
  ```
- Check the file output (can be loaded into Pandas/Spark):
  ```
  data/output/cleaned_fleet_data.parquet
  ```

---

## Frequently Asked Questions

**Q: What is “pushdown filtering”?**  
A: It means Spark uses a SQL subquery in JDBC to fetch only what you need (no nulls, no “bad” rows), reducing the amount of data transferred and processed.

**Q: Can I use this with other databases?**  
A: Yes, any JDBC-compatible DB; you’d only need to change the driver and properties in `utils.py`.

**Q: What if I want to write to AWS S3 instead of local Parquet?**
A: No code changes needed in your ETL logic — only change the file path to use `s3a://bucket_name/...` and include the Hadoop AWS connector JAR and credentials.  
(This can be easily added later.)

---

## How the Mechanism Works (utils.py Comments)

- `get_spark_session()` creates/configures Spark and adds the necessary JDBC driver, enabling Spark → PostgreSQL reads/writes.
- `get_jdbc_properties()` centralizes all Postgres connection config (URL/credentials), so you only change details in one place.
- Both are imported into your ETL main script and abstract all Spark/DB connectivity setup for simplicity and reusability.

---

## Troubleshooting

- **OutOfMemory errors:**
  - Lower `numPartitions` or batch sizes, or run on a more powerful machine.
- **Cannot find JDBC driver:**
  - Double-check your path in `utils.py` to `lib/postgresql-42.7.3.jar`
- **Database connection refused:**
  - Check Postgres is running, credentials/port match, DB allows connections.

---

## Next Steps & Extending

- Partition Parquet files by `event_type` or `date` for BI analytics
- Schedule ETL with cron or Airflow for recurring processing
- Easily plug in S3/Glue Data Catalog for cloud scale

---

**This ETL pattern is ready to scale: adjust paths/DBs as needed, and you'll be ready for larger or cloud-based data pipelines with minimal changes.**
