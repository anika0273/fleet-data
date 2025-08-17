# Metric Calculation & Predictive Analytics

(Advanced, modular, batch jobs on top of streaming outputs)

## A self-contained analytics module that:

- Loads cleaned/processed events (e.g., from public.fleet_stream_processed) from PostgreSQL.

- Computes windowed metrics (5-minute tumbling windows per vehicle): For each vehicle, it divides data into fixed 5-minute intervals (“tumbling windows”). It calculates metrics for each window—such as average speed, braking frequency, estimated distance traveled, and a composite risk score summarizing risky driving behavior..

- Unsupervised ML for Clustering Behavioral Patterns: The module builds feature vectors from those metrics and trains a KMeans clustering model to discover typical behavior clusters (e.g., “normal driving” vs “aggressive driving”). Each window is assigned to a cluster, and the distance to cluster centroid serves as an anomaly score—high distance means more unusual (possibly risky) behavior.

- Anomaly Detection: Using the cluster assignments and distances, the code flags outlier windows as “anomalous,” making it easier to spot unsafe or unexpected driving patterns for further review.

- Persists Results: It writes aggregated metrics and anomaly/cluster labels back into PostgreSQL. These results can then feed dashboards, BI tools, or risk-monitoring systems.

- Ships with unit tests, argparse CLI, and env-driven config.

- This module is batch (not streaming). It’s meant to be re-run safely (hourly/daily) on top of the data the streaming ETL continuously appends.

# Run metrics.py

```bash
spark-submit \
  --jars /Users/owner/Desktop/fleet-data/lib/postgresql-42.7.3.jar \
  metrics/metrics.py
```

# Verify Metrics in PostgreSQL

```sql
-- View recent windows
SELECT *
FROM telemetry_metrics_vehicle_5m
ORDER BY window_end DESC, vehicle_id
LIMIT 50;

-- Vehicles with highest overspeed rate in last hour
SELECT vehicle_id,
       AVG(overspeed_rate) AS avg_overspeed_rate,
       COUNT(*) AS windows
FROM telemetry_metrics_vehicle_5m
WHERE window_end > NOW() - INTERVAL '1 hour'
GROUP BY vehicle_id
ORDER BY avg_overspeed_rate DESC
LIMIT 20;

-- Vehicles with most GPS missing ratio today
SELECT vehicle_id,
       AVG(gps_missing_ratio) AS avg_gps_missing_ratio
FROM telemetry_metrics_vehicle_5m
WHERE window_end::date = CURRENT_DATE
GROUP BY vehicle_id
ORDER BY avg_gps_missing_ratio DESC
LIMIT 20;
```
