# Fleet Telemetry Synthetic Data Generator

## Overview

This project creates **realistic but synthetic fleet telemetry data** to simulate connected vehicle (IoT) scenarios — e.g., GPS tracking, sensor readings, and event alerts from vehicles in a fleet.

It generates **large-scale, messy, and varied** data that closely mirrors the complexity of real-world data streams, and uploads it directly into a **PostgreSQL** database for analysis or machine learning.

### Why Synthetic?

- **Real-world fleet telemetry** data is often private and sensitive (personal information, geolocation).
- **Synthetic data** allows testing ETL pipelines, dashboards, and analytics **without privacy concerns**.
- We specifically make this synthetic data "dirty" with **outliers, missing values, and inconsistencies** to mimic what analysts face in real deployments.

---

## Features & Data Fields with Design Rationale

Each generated record is one observation (event) from a fleet vehicle.

| Field                      | Data Type     | Generation Logic                                                                                | Why This Approach?                                                                                                          |
| -------------------------- | ------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `vehicle_id`               | `VARCHAR(20)` | Chosen from a pre-generated pool of N unique license plates. Fleet size = 500 vehicles.         | Real fleets have repeated vehicles over time, not a new ID for each row. Limiting IDs enables **vehicle history analysis**. |
| `timestamp`                | `TIMESTAMP`   | Random datetime within the last 30 days.                                                        | Recent data ensures relevance for time-based analytics and keeps file size manageable.                                      |
| `latitude`, `longitude`    | `FLOAT`       | Random inside NYC bounding box (LAT 40.4774–40.9176, LON -74.2591––73.7004). ~0.5% set to NULL. | Constraining coordinates to NYC gives realistic spatial clustering. Occasional NULL simulates GPS dropouts/loss of signal.  |
| `speed`                    | `FLOAT`       | 0–130 km/h normally. ~0.2% chance of unrealistic 150–250 km/h "outliers".                       | Max normal speed covers highways; outliers mimic real corrupted readings or unit errors.                                    |
| `event_type`               | `VARCHAR(30)` | Derived from boolean event flags based on **priority order** (e.g., collision > lane change).   | Vehicles often have multiple simultaneous events; the highest-priority is most important to downstream systems.             |
| `braking_event`            | `BOOLEAN`     | ~7% base chance, 15% during rush hours (07–09 & 16–18).                                         | Dense traffic increases braking frequency; rush hour chosen from real commute patterns.                                     |
| `collision_alert`          | `BOOLEAN`     | 0.5% base chance, doubled in rain/fog/snow.                                                     | Accidents are rare but more likely in bad conditions; probability tuned to keep events rare but visible.                    |
| `lane_change_event`        | `BOOLEAN`     | ~10% chance.                                                                                    | Higher than collision but still infrequent; matches typical telematics reports.                                             |
| `harsh_acceleration_event` | `BOOLEAN`     | ~3% chance.                                                                                     | Rare but common in aggressive driving data.                                                                                 |
| `sensor_fault_event`       | `BOOLEAN`     | ~0.2% chance.                                                                                   | Hardware failures should be rare; tuned for small realistic proportion.                                                     |
| `network_delay_event`      | `BOOLEAN`     | ~0.1% chance.                                                                                   | Low probability reflects occasional transmission delays in IoT networks.                                                    |
| `gps_loss_event`           | `BOOLEAN`     | ~0.1% chance.                                                                                   | Minimal but realistic GPS signal drops.                                                                                     |
| `weather`                  | `VARCHAR(20)` | Random choice from sunny, rainy, foggy, snowy, cloudy.                                          | Found in real telematics records; influences probabilities for braking/collision.                                           |
| `road_type`                | `VARCHAR(20)` | Random: highway, city, rural.                                                                   | Provides context for speed, risk, event type.                                                                               |
| `traffic_density`          | `VARCHAR(20)` | Random: low, medium, high.                                                                      | Context indicator, combined with time-of-day for analysis.                                                                  |
| `hour_of_day`              | `SMALLINT`    | Extracted from timestamp.                                                                       | Used for rush-hour checks, temporal analysis.                                                                               |
| `sensor_battery`           | `FLOAT`       | 20–100%, with ~0.3% chance >100%.                                                               | Corrupted power readings exist in real IoT devices.                                                                         |
| `sensor_signal_strength`   | `FLOAT`       | Uniform 1–5 scale.                                                                              | Represents link quality, spoofed from real device metrics.                                                                  |
| `risk_label`               | `BOOLEAN`     | False if collision/sensor fault else True. 0.2% inverted intentionally.                         | Acts as binary classification label; inversion simulates bad labeling in real world.                                        |

---

## How Data Dirtiness Is Introduced

1. **Missing GPS coordinates** — Simulates telemetry dropouts.
2. **Outliers in speed and battery** — Mimics sensor malfunctions or unit mismatches.
3. **Contradictory labels** — Risk label sometimes doesn't match events, simulating mislabeled training data.
4. **Environmental influence** — Rain/fog/snow increases braking/collision probabilities.
5. **Repeated vehicles over time** — Enables time-series analysis.
6. **Random noise** — Avoids perfectly clean distributions.

---

## Requirements

- **Python** ≥ 3.8
- **PostgreSQL** ≥ 12
- **pgAdmin** (optional GUI)

Python libraries:

```bash
pip install psycopg2-binary pandas faker
```

# Database Setup

## 1. Verify PostgreSQL

```bash
pg_isready -h localhost -p 5433
```

## 2. Create Database

```bash
CREATE DATABASE fleet_db;
```

## 1. Create Table

```sql
CREATE TABLE fleet_data (
    vehicle_id VARCHAR(20),
    timestamp TIMESTAMP,
    latitude FLOAT,
    longitude FLOAT,
    speed FLOAT,
    event_type VARCHAR(30),
    braking_event BOOLEAN,
    collision_alert BOOLEAN,
    lane_change_event BOOLEAN,
    harsh_acceleration_event BOOLEAN,
    sensor_fault_event BOOLEAN,
    network_delay_event BOOLEAN,
    gps_loss_event BOOLEAN,
    weather VARCHAR(20),
    road_type VARCHAR(20),
    traffic_density VARCHAR(20),
    hour_of_day SMALLINT,
    sensor_battery FLOAT,
    sensor_signal_strength FLOAT,
    risk_label BOOLEAN
);
```

## Script Configuration

Update database credentials in `connect_to_postgres()`:

```python
host = "localhost"
port = "5433"
database = "fleet_db"
user = "postgres"
password = "1234"
```

## Running the Script

```bash
python generate_data.py
```

### Flow:

1. Generate num_records synthetic entries.
2. Create/verify fleet_data table.
3. Bulk insert into PostgreSQL via execute_values.

## VerifyingData Load

```sql
SELECT COUNT(*) FROM fleet_data;
SELECT * FROM fleet_data LIMIT 10;
```

### Check for:

- NYC coordinates
- Repeated vehicle IDs
- Random NULLs/outliers
- Variety in event flags

## Performance Notes

- **100k rows** load in seconds with batch inserts.
- Optional indexing:

```sql
CREATE INDEX idx_vehicle_id ON fleet_data(vehicle_id);
CREATE INDEX idx_timestamp ON fleet_data(timestamp);
```

## Why these probabilities?

- Collision: Rare (0.5%), matching real-world frequency.

- Braking: Higher during rush hours, per traffic data.

- Outliers: <1%, enough for anomaly detection testing.

- Weather effects: Bad weather ~doubles collision/braking.

- Fleet size: 500 vehicles for variety + repeated history.

## Possible Data Extension

- Change GPS bounding box for other cities.

- Adjust fleet_size / num_records.

- Stream data instead of batch inserts.

- Add maintenance logs, fuel use, driver IDs.
