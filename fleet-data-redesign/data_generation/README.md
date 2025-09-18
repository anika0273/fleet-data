# Synthetic Fleet and Supply Chain Telemetry Data Generator

---

## Overview

This Python-based synthetic data generator produces a comprehensive dataset simulating realistic fleet telemetry records integrated with detailed supply chain shipment events and delays. It is designed to reflect the complex interplay between vehicles, drivers, devices, trips, and shipment logistics operations, enabling research, machine learning, and operational analytics in fleet management and supply chain domains.

---

## Motivation and Design Rationale

Real-world fleet and supply chain data is often incomplete, highly variable, and costly to obtain at scale for experimental or ML model development. This generator creates controlled yet realistic synthetic records simulating critical operational, behavioral, and logistical processes to facilitate:

- **Driver behavior intervention analysis** (reducing risky events through training and alerts)
- **Route optimization effects** on fuel consumption and efficiency
- **Predictive maintenance impact** on vehicle downtime and costs
- **Real-time feedback influence** on safety metrics
- **Technology adoption benefits** from newer telematics devices
- **Supply chain ETA predictions and delay root cause investigations**

---

## Detailed Explanation of Columns and Logic

Each column is carefully crafted based on domain expertise and literature to ensure realism and interpretability.

### Identifiers & Temporal

- **vehicle_id**: Randomly generated license plates to uniquely identify vehicles in a fleet sized to 500 by default.
- **driver_id**: Numeric IDs for drivers in a pool of 400, allowing driver-specific analytics.
- **route_id**: Simple coded route identifier representing trip paths.
- **shipment_id**: Unique shipment identifiers to link supply chain details.
- **timestamp**: Random datetime within past 30 days marking each trip snapshot.
- **hour_of_day**: Extracted hour to simulate time-of-day effects including rush hours; allows temporal traffic/delay modeling.

### Geographic and Environmental Context

- **latitude & longitude**: Random coordinates within NYC bounding box to simulate real vehicle GPS locations, including 0.5% dropout to mimic GPS loss.
- **trip_weather**: Selected from `['sunny', 'rainy', 'foggy', 'snowy', 'cloudy']` with realistic probabilities reflecting typical urban weather distributions; impacts event rates, fuel efficiency, and delay probabilities.
- **rush_hour**: Boolean flag indicating peak traffic periods (morning 7–9am and evening 4–6pm).
- **road_type & traffic_density**: Road (highway, city, rural) and traffic (low, medium, high) sampled with realistic weightings. These strongly influence speed, idle, fuel, and event risks.

### Vehicle Characteristics

- **vehicle_type**: Categories `car`, `van`, `truck` with allocation weights reflecting typical fleet mixes.
- **vehicle_age_years**: Integer 0–12 years simulating wear and degradation, influencing fuel consumption and breakdown risk.
- **device_generation**: IoT device generation (1, 2, or 3) representing evolution in telemetry tech, impacting data latency, GPS accuracy, and loss rates.
- **device_cost_usd**: Simulated device purchase costs correlated with generation.
- **maintenance_type**: Predictive vs. scheduled maintenance with impacts on downtime and failure probability.

### Driver Attributes

- **driver_experience_years**: Integer 0–20 years simulating driver skill/experience.
- **driver_training**: Binary trained vs untrained status, affecting risk of harsh accelerations and collisions.

### Route and Trip Features

- **optimized_route_flag**: Boolean indicating if route optimization product is used; influences reduced distance and idle times.
- **distance_traveled_km**: Sampled based on road type Gaussian distributions (e.g., highways average longer distances).
- **speed**: Sampled speed following distributions per road/traffic conditions, with rare extreme outliers to simulate unsafe driving.
- **idle_time_minutes**: Reflects idling duration as function of road and traffic congestion.
- **fuel_rate_l_per_100km**: U-shaped speed effect curve combined multiplicatively with weather, traffic, device generation, and vehicle base fuel factors.
- **fuel_consumption_liters**: Derived from distance, idle burning, blended with small noise to simulate variation.

### Device & Network Metrics

- **sensor_battery**: Percent battery level with rare >100% calibration outliers to mimic sensor errors.
- **sensor_signal_strength**: Signal strength scaled by device gen with random noise.
- **data_latency_ms, gps_accuracy_meters, packet_loss_rate**: Sampled normally per device gen, including occasional spikes mimicking network degradation or GPS disruptions.

### Driving and System Events

- Event flags (Boolean) denoting braking, collisions, lane changes, harsh acceleration, sensor faults, network delay, GPS loss.
- Probabilities modified by driver training, weather, traffic, and intervention active.
- Primary event type recorded as string priority (collision highest).
- Collision or sensor fault events increase breakdown probability.
- Harsh acceleration reduced by intervention active on trip.

### Maintenance & Breakdown Outcomes

- Breakdown event probability dependent on vehicle age, maintenance type, and event flags.
- Downtime and maintenance cost simulated with Gaussian noise and cost multipliers reflective of predictive vs scheduled regimes.
- Time since last maintenance sampled with lower values for predictive group.

### Supply Chain and Delay Features

- **carrier_name, warehouse_origin, warehouse_destination**: Discrete categorical variables representing shipment origin/destination and transport company, allowing lane-specific delay analytics.
- **supplier_id, supplier_region**: Upstream supplier codes and regions, enabling supplier influence analysis.
- **carrier_service_level**: Reflects shipping priority class, affecting expected delays.
- **planned_departure_time, planned_arrival_time**: Scheduled times for modeling ETA.
- **actual_departure_time, actual_arrival_time**: Derived from planned plus additive delays such as loading.
- **customs_hold_flag & duration**: Randomly active customs delay (4% chance), with durations 20-480 minutes.
- **weather_delay_flag & duration**: Conditional on adverse weather, 15% chance if snowy, rainy, or foggy.
- **traffic_delay_flag & duration**: Dependent on high traffic, 30% chance, durations between 5-90 minutes.
- **loading_delay_flag & duration**: Simulates peak hour warehouse loading delays at 18% probability.
- **delay_minutes**: Aggregate delay for ETA adjustment.
- **shipment_status**: In-transit, late (over 30 minutes delay), or exception.
- **late_shipment_cost_usd**: Simulated penalty cost combining fixed and delay-proportional amounts.

### Labeling and Intervention

- **intervention_active**: Flag simulating if real-time alerts/training were active, lowering risky events.
- **risk_label**: Boolean target for supervised learning indicating if trip was risky, based on collision, fault or breakdown.

---

## Reasons Behind Ranges and Distributions

- **Vehicle age, driver experience, and device generation** ranges mirror typical fleet demographic data from published industry insights (e.g., average fleet truck age 6-8 years, IoT device upgrade cycles).
- **Speed and distance distributions** are parameterized to typical urban highway and city driving behaviors, adjusted for traffic congestion to reflect realism.
- **Fuel rates use known vehicular fuel curves with speed efficiency dips and weather impact coefficients** derived from transportation engineering literature.
- **Event probabilities** incorporate base rates multiplied by conditional risk factors (weather, driver training) consistent with fleet safety statistics.
- **Delay probabilities and durations** mirror operational supply chain stats: customs holds (4%), weather delays (15%), traffic-based congestion (30%), and loading times aligning with rush hours.
- **Shipment cost penalties model financial impacts from industry averages and logistic cost structures.**

---

## Strengths of the Synthetic Data

- **Domain-grounded:** Parameters drawn from literature and industry intuition to ensure practical realism.
- **Multi-dimensional:** Captures multi-entity relationships (vehicle, driver, shipment) for rich analysis.
- **Noise and outliers:** Planned noise and rare irregular values resemble real-world sensor and operational variability.
- **Intervention effect simulation:** Enables causal and experimental design analyses.
- **Comprehensive labels and KPIs:** Enables multi-task ML such as risk prediction, ETA, and cost estimation.

---

## Limitations and Real-World Data Challenges

- **Simplified geography:** Single-city bounding box limits spatial diversity.
- **Limited longitudinal dynamics:** Static driver and vehicle attributes miss behavioral changes over time.
- **Sparse shipment hierarchy:** Flat shipment-trip snapshots omit multi-leg or multi-modal complexities.
- **Minimal missingness:** Real data typically has more missing or corrupted records requiring non-trivial cleaning.
- **Simplified delay modeling:** Additive delays lack complex interactions and time-varying dynamics.

---

## Handling Real-World Data

- Deploy robust cleaning pipelines addressing missing data, duplicates, and inconsistencies.
- Use advanced imputation or noise-tolerant ML methods to handle unstructured sensor outages.
- Aggregate multi-source and multi-stage shipment telemetry hierarchically.
- Integrate external real-time or historical data sources for environment features.
- Retrain and calibrate models to handle concept drift in dynamic fleets.

---

## Future Feature Enhancements

- Expand geospatial simulation beyond single city.
- Incorporate driver behavioral time series.
- Simulate multi-leg shipment route decompositions.
- Model variable sensor reliability over time.
- Integrate real-world weather and traffic feeds.
- Develop probabilistic delay forecasting incorporating weather and customs complex interactions.

---

## Running the Generator and Storing Data

- The main entrypoint is `generate_fleet_data(num_records, fleet_size, driver_pool)` returning a pandas DataFrame.
- The included `main()` script generates 100,000 records by default and persists to a PostgreSQL instance.
- Database schema matches DataFrame columns exactly; aggregation, validation, and indexing should be performed post-ingestion.

---

## Column Summary Table

| Column Name                      | Description                                        | Value Range / Distribution                       |
| -------------------------------- | -------------------------------------------------- | ------------------------------------------------ |
| vehicle_id                       | Unique vehicle license plate                       | Random alphanum string                           |
| driver_id                        | Unique driver ID                                   | "DRV0000" to "DRV0399" (driver_pool size)        |
| route_id                         | Trip route code                                    | Random "RTE100" to "RTE999"                      |
| timestamp                        | Trip record datetime                               | Random past 30 days                              |
| shipment_id                      | Unique shipment/order ID                           | "SHP100000" to "SHP999999"                       |
| carrier_name                     | Carrier company                                    | Random from four fixed options                   |
| warehouse_origin                 | Shipment origin warehouse                          | Random from four fixed options                   |
| warehouse_destination            | Shipment destination warehouse                     | Origin distinct from destination                 |
| supplier_id                      | Supplier numeric ID                                | Integer 1 - 200                                  |
| supplier_region                  | Supplier region                                    | One of five discrete options                     |
| carrier_service_level            | Shipping priority class                            | Standard, Expedited, WhiteGlove                  |
| planned_departure_time           | Scheduled shipment departure time                  | Calculated as `timestamp` minus 1–12 hours       |
| actual_departure_time            | Real shipment departure time                       | Planned plus loading delay                       |
| planned_arrival_time             | Scheduled shipment arrival time                    | Planned departure plus 2–9 hours                 |
| actual_arrival_time              | Real shipment arrival time                         | Planned arrival plus sum of delays               |
| eta_minutes                      | ETA predicted duration (mins)                      | 120–540+                                         |
| delay_minutes                    | Aggregate shipment delay in minutes                | Sum of customs, weather, traffic, loading delays |
| customs_hold_flag                | Customs delay (boolean)                            | ~4% probability                                  |
| customs_hold_duration_min        | Customs hold time (minutes)                        | 20–480 when active                               |
| weather_delay_flag               | Weather-related delay indicator                    | ~15% probability on rainy/snowy/foggy            |
| weather_delay_minutes            | Minutes delayed by weather                         | 10–120 when active                               |
| traffic_delay_flag               | Traffic-related delay flag                         | ~30% when traffic high                           |
| traffic_delay_minutes            | Minutes delayed by traffic                         | 5–90 when active                                 |
| loading_delay_flag               | Loading/unloading delay flag                       | ~18% during morning/evening rush hours           |
| loading_delay_minutes            | Minutes delayed loading                            | 6–45 when active                                 |
| shipment_status                  | Shipment state ("in_transit", "late", "exception") | Derived from delay_minutes                       |
| late_shipment_cost_usd           | Cost penalty for delays                            | $25+ scaled by delay_minutes                     |
| latitude                         | Vehicle GPS latitude                               | NYC bounding box floating value                  |
| longitude                        | Vehicle GPS longitude                              | NYC bounding box floating value                  |
| hour_of_day                      | Integer trip hour                                  | 0–23                                             |
| trip_weather                     | Weather at trip timestamp                          | Five categories as above                         |
| road_type                        | Road classification (highway, city, rural)         | Fixed categorical                                |
| traffic_density                  | Local traffic density (low, medium, high)          | Weighted by rush hour status                     |
| rush_hour                        | Rush hour flag                                     | Boolean                                          |
| vehicle_type                     | Vehicle class                                      | car, van, truck                                  |
| vehicle_age_years                | Vehicle age (years)                                | 0–12                                             |
| device_generation                | IoT device generation                              | 1, 2, or 3                                       |
| device_cost_usd                  | IoT device purchase cost                           | $200–$800                                        |
| driver_experience_years          | Years of driver experience                         | 0–20                                             |
| driver_training                  | Training status                                    | trained/untrained                                |
| optimized_route_flag             | Route optimization usage                           | Boolean                                          |
| speed                            | Vehicle speed (km/h)                               | 0–240 with rare outliers                         |
| distance_traveled_km             | Distance driven in trip segment                    | Road-dependent Gaussian                          |
| idle_time_minutes                | Idling time (minutes)                              | Road and traffic influenced                      |
| fuel_consumption_liters          | Calculated fuel consumption                        | Combined idle and distance-based burn            |
| fuel_rate_l_per_100km            | Fuel consumption rate (liters/100km)               | Curved and conditionally modulated               |
| sensor_battery                   | Sensor battery level (%)                           | 20–100+ rare outliers                            |
| sensor_signal_strength           | Signal quality indicator                           | 1.2–5                                            |
| data_latency_ms                  | Data latency (ms)                                  | 65–180+ rare spikes                              |
| gps_accuracy_meters              | GPS signal accuracy (meters)                       | 5–14+ rare outliers                              |
| packet_loss_rate                 | Network packet loss rate                           | 0.001–0.03                                       |
| braking_event                    | Boolean flag for braking                           | Probability based on context                     |
| collision_alert                  | Collision event flag                               | Rare, context influenced                         |
| lane_change_event                | Lane change event flag                             | Common, mostly highway                           |
| harsh_acceleration_event         | Exposure to harsh acceleration                     | Driver training and intervention affected        |
| sensor_fault_event               | Sensor fault boolean                               | Higher on older devices                          |
| network_delay_event              | Network delay boolean                              | Higher on older devices                          |
| gps_loss_event                   | GPS loss boolean                                   | Higher on older devices                          |
| event_type                       | Dominant event label                               | Priority order collision > fault > etc.          |
| maintenance_type                 | Maintenance scheduling model                       | predictive/scheduled                             |
| time_since_last_maintenance_days | Days since last maintenance                        | Shorter with predictive maintenance              |
| breakdown_event                  | Mechanical breakdown                               | Based on age, faults, collisions                 |
| downtime_hours                   | Hours offline due to breakdown                     | Longer for collision-induced failures            |
| maintenance_cost_usd             | Cost of maintenance including downtime             | Varied by regime and downtime                    |
| intervention_active              | Real-time driver intervention active               | Boolean                                          |
| risk_label                       | Outcome label: safe vs risky trip                  | Boolean with noise for realism                   |

---

## Strengths and Weaknesses

**Strengths:**

- Realistic, multi-layer correlated dataset supporting safety, efficiency, maintenance, and supply chain analytics.
- Configurable fleet and shipment sizes for scaling experiments.
- Incorporation of domain knowledge in probability tuning, event correlations, and delay cause modeling.
- Designed to enable classical and ML predictive modeling, A/B testing, and root-cause analysis.

**Weaknesses:**

- Simplified spatial/temporal model centered on a single metro region.
- Minimalisy hierarchical shipment vs trip decomposition.
- Underrepresentation of real-world noisy and missing data complexity.
- Static driver and vehicle states ignoring longitudinal behavior evolution.
- Simplistic additive delay modeling missing interactions and dynamic feedback.

---

## Recommendations for Real-World Data Use

- Implement sophisticated data validation, cleaning, and imputation pipelines.
- Augment with hierarchical and time series modeling to capture shipment progress.
- Integrate external weather and traffic data sources for environmental context.
- Use robust ML pipelines tolerant to data quality issues and evolving patterns.

---

## Future Directions

- Add fine-grained multi-leg shipment modeling.
- Simulate sensor failure and recovery dynamics over time.
- Incorporate driver behavioral fatigue and other temporal dynamics.
- Model multi-modal logistics, regulatory inspections, and customs process delays.
- Automate dynamic weather, traffic, and event feed simulation.

---

## Getting Started

1. Run `generate_fleet_data()` for synthetic DataFrame generation.
2. Use `main()` to generate and optionally save data to PostgreSQL.
3. Customize parameters for fleet size, driver pool, or record counts.
4. Analyze data for A/B experiments, ML model training, or dashboarding.

---

This README makes the dataset and generator fully understandable for researchers and engineers aiming to build scalable, realistic fleet and supply chain analytics using synthetic data.

---

_End of README_
