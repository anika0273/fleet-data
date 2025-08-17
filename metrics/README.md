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
