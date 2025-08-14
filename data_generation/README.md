# Synthetic Fleet Data Generation

This module generates a **rich, realistic synthetic dataset** mimicking vehicle fleet operational data, aimed to reflect real-world complexity and edge cases.

## Key Enhancements and Why They Matter

### 1. **Multi-event Simulation**

- Vehicles can experience multiple simultaneous events such as braking during high traffic or sensor faults alongside GPS loss.
- Enables modeling real complex interactions rather than independent, simple events.

### 2. **Contextual Features**

- Variables like **weather** (rainy, foggy, sunny), **road type** (city, highway, rural), and **traffic density** influence event probabilities.
- Reflects how environmental factors impact driving behavior and data collection.

### 3. **Temporal Awareness**

- Events’ probabilities vary based on **time of day** (rush hour vs. late night), adding natural patterns.
- Helps in realistic temporal analytics and forecasting.

### 4. **Sensor Health Metrics**

- Added fields like **sensor battery** and **signal strength** mimic real hardware conditions.
- Critical for detecting sensor degradation or communication issues.

### 5. **Edge Cases and Rare Events**

- Introduces **rare but critical conditions** such as GPS signal loss & network delays.
- Important for testing robustness of downstream processing and alerting systems.

### 6. **Ground Truth Labels**

- Each record is tagged with a **risk label** (safe/risky), simulating target variables for machine learning.
- Prepares dataset for predictive modeling or anomaly detection.

### 7. **Large Scale**

- Scalable generation with adjustable record count (e.g., 100,000+ records) to demonstrate pipeline capabilities with realistic data volume.

---

## How To Use This Module

- Run `generate_data.py` to produce data and save directly to your PostgreSQL database.
- Use this dataset for both batch and streaming simulation in later pipeline stages.
- Customize parameters (record counts, probabilities) to tailor data complexity.
- Extend by adding new event types or sensor metrics as needed.

---

This approach demonstrates an advanced understanding of:

- Real-world synthetic data simulation for autonomous vehicle fleets,
- Preparation for downstream machine learning and anomaly detection,
- Designing data pipelines that handle complex, multi-dimensional datasets.

---

_References:_ Inspired by best practices in Tesla’s FSD data simulation and modern data engineering principles.

---

## Next Steps: Streaming Data Guidance (for later implementation)

While this module generates batch data, streaming data ingestion will simulate **continuous real-time vehicle telemetry** by emitting this data, for example, via Apache Kafka topics.

---

**How to use the generated data for streaming:**

- Package records into JSON messages and send them to a Kafka topic.
- Develop a Kafka producer script (in Python) that reads from this dataset (or generates on-the-fly) and streams events in intervals.
- On the consumer side, Spark Structured Streaming or a similar engine subscribes to Kafka topics, processes incoming data in near real-time, and persists metrics.
- This setup mirrors real autonomous fleet scenarios where live telemetry is ingested continuously.

---

Would you like me to help build the Kafka producer script and show how to build a basic Spark streaming job next? Or help you integrate this data generation neatly into your project repo and add the README?
