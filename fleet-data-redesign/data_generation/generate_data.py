# data_generation/generate_data.py

# Check if PostgreSQL run: pg_isready -h localhost -p 5433

import random
import math
import pandas as pd
from faker import Faker
import psycopg2
from typing import Optional
import psycopg2.extras as extras
from datetime import datetime

import sys
import os
"""
Synthetic Fleet Telemetry Generator (Realistic, Correlated, ML-Ready)
--------------------------------------------------------------------
Covers 5 portfolio analyses:
1) Driver interventions (training/alerts) → risky events
2) Route optimization → fuel consumption
3) Predictive maintenance → downtime
4) Real-time feedback → harsh events & wear
5) Device generation → reliability & data quality

Design choices:
- Hierarchical entities (vehicle, driver, trip/route) with stable attributes per entity.
- Correlated probabilities (weather/traffic/rush hour worsen events; training/feedback reduce them).
- Fuel model combines distance + idle burn; route optimization reduces distance & idle.
- Predictive maintenance reduces breakdown/downtime & cost.
- Newer devices improve GPS accuracy, latency, packet loss; fewer network issues.
- Inject light noise, rare nulls, and controlled outliers for realism.

Notes on labels:
- `risk_label` is a BOOLEAN where True = SAFE, False = RISKY (collision/sensor fault/breakdown). A tiny label noise is injected.
"""

fake = Faker()

# ------------------------------------------------------------------
# Region / GPS settings (NYC bounding box)
# ------------------------------------------------------------------
LAT_MIN, LAT_MAX = 40.4774, 40.9176
LON_MIN, LON_MAX = -74.2591, -73.7004


def generate_gps_coordinates():
    """Generate GPS coordinates within the bounding box. 0.5% dropout -> (None, None)."""
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    if random.random() < 0.005:  # missingness to simulate GPS loss in raw coords
        return None, None
    return round(lat, 6), round(lon, 6)


# ------------------------------------------------------------------
# Core generator
# ------------------------------------------------------------------

def generate_fleet_data(num_records: int = 100_000,
                         fleet_size: int = 500,
                         driver_pool: int = 400):
    """
    Generate synthetic fleet telemetry with realistic correlations and enough signal for ML.

    Row = one trip snapshot / record (not every second). Each record has:
      - Vehicle features (type, age, device gen, maintenance type)
      - Driver features (experience, training)
      - Route context (road, traffic, weather, rush hour, optimization flag)
      - Events (braking, harsh accel, collision, lane change, device/network issues)
      - KPIs (distance, idle, fuel consumption, downtime, maintenance cost)

    Returns: pandas.DataFrame
    """

    rng = random.Random(42)  # deterministic-ish reproducibility

    # ---------------- Vehicle master data (stable per vehicle) ----------------
    vehicle_types = ["car", "van", "truck"]
    vehicles = {}
    for _ in range(fleet_size):
        vid = fake.license_plate()
        vtype = rng.choices(vehicle_types, weights=[0.45, 0.25, 0.30])[0]
        age_years = rng.randint(0, 12)  # age impacts fuel & breakdown risk
        device_gen = rng.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
        maintenance_type = rng.choices(["predictive", "scheduled"], weights=[0.6, 0.4])[0]
        # baseline fuel efficiency by vehicle type + aging penalty (L/100km)
        base_rate = {
            "car": 7.5,
            "van": 12.0,
            "truck": 28.0
        }[vtype] + age_years * {
            "car": 0.08,
            "van": 0.15,
            "truck": 0.25
        }[vtype]
        # device cost (not essential, but useful for ROI comparisons)
        device_cost_usd = {
            1: rng.uniform(200, 350),
            2: rng.uniform(351, 550),
            3: rng.uniform(551, 800)
        }[device_gen]
        # route optimization access at the vehicle level (adoption differences)
        route_opt_access = rng.random() < 0.6  # 60% vehicles have optimization product enabled

        vehicles[vid] = {
            "vehicle_type": vtype,
            "vehicle_age_years": age_years,
            "device_generation": device_gen,
            "maintenance_type": maintenance_type,
            "base_fuel_rate_l_per_100km": base_rate,
            "device_cost_usd": round(device_cost_usd, 2),
            "route_opt_access": route_opt_access,
        }

    # ---------------- Driver master data (stable per driver) ------------------
    drivers = {}
    for i in range(driver_pool):
        did = f"DRV{i:04d}"
        exp = rng.randint(0, 20)
        trained = rng.random() < 0.65
        drivers[did] = {
            "driver_experience_years": exp,
            "driver_training": "trained" if trained else "untrained",
        }

    # ---------------- Helper functions for correlated sampling ----------------
    def sample_context(ts_hour):
        weather = rng.choices(['sunny', 'rainy', 'foggy', 'snowy', 'cloudy'],
                              weights=[0.5, 0.18, 0.08, 0.06, 0.18])[0]
        road = rng.choices(['highway', 'city', 'rural'], weights=[0.45, 0.4, 0.15])[0]
        # rush hour modifies traffic distribution
        rush_hour = (7 <= ts_hour <= 9) or (16 <= ts_hour <= 18)
        if rush_hour:
            traffic = rng.choices(['low', 'medium', 'high'], weights=[0.15, 0.45, 0.40])[0]
        else:
            traffic = rng.choices(['low', 'medium', 'high'], weights=[0.30, 0.50, 0.20])[0]
        return weather, road, traffic, rush_hour

    def sample_speed(road, traffic):
        # Base speed by road, then drag down if high traffic
        mu, sigma = {
            'highway': (105, 12),
            'city': (48, 10),
            'rural': (65, 12)
        }[road]
        speed = rng.gauss(mu, sigma)
        if traffic == 'high':
            speed -= rng.uniform(8, 20)
        speed = max(0, min(speed, 160))
        # Rare extreme outlier
        if rng.random() < 0.002:
            speed = rng.uniform(160, 240)
        return round(speed, 2)

    def sample_distance_km(road):
        # Trip-level distance distribution by road type
        if road == 'highway':
            dist = max(5, rng.gauss(45, 18))
        elif road == 'city':
            dist = max(2, rng.gauss(12, 5))
        else:
            dist = max(3, rng.gauss(25, 10))
        return round(dist, 1)

    def sample_idle_minutes(road, traffic):
        base = {
            'highway': rng.uniform(1, 4),
            'city': rng.uniform(4, 12),
            'rural': rng.uniform(2, 6)
        }[road]
        # traffic penalty
        base *= {'low': 0.8, 'medium': 1.0, 'high': 1.6}[traffic]
        return round(base, 1)

    def fuel_rate_l_per_100km(vehicle, speed, weather, traffic, device_gen):
        # U-shaped speed effect with minimum near ~65 km/h
        speed_factor = 1 + 0.003 * abs(speed - 65)
        weather_factor = {
            'sunny': 1.0, 'cloudy': 1.0,
            'foggy': 1.04, 'rainy': 1.10, 'snowy': 1.15
        }[weather]
        traffic_factor = {'low': 0.95, 'medium': 1.0, 'high': 1.12}[traffic]
        device_factor = {1: 1.00, 2: 0.99, 3: 0.97}[device_gen]  # newer slightly better
        base = vehicle['base_fuel_rate_l_per_100km']
        rate = base * speed_factor * weather_factor * traffic_factor * device_factor
        # small noise
        rate *= rng.uniform(0.97, 1.03)
        return max(3.5, round(rate, 2))

    def idle_burn_lpm(vehicle_type):
        return {'car': 0.04, 'van': 0.06, 'truck': 0.10}[vehicle_type]

    def route_optimization_effect(has_access):
        # if the vehicle has access, 80% of trips use optimization; otherwise 20% adopt
        if has_access:
            use_flag = rng.random() < 0.8
        else:
            use_flag = rng.random() < 0.2
        # distance reduced 5–12%, idle reduced 10–30% when used
        dist_mult = rng.uniform(0.88, 0.95) if use_flag else 1.0
        idle_mult = rng.uniform(0.70, 0.90) if use_flag else 1.0
        return use_flag, dist_mult, idle_mult

    def device_metrics(device_gen):
        # Better gen → stronger signal, lower latency, better GPS accuracy, lower loss
        sig = rng.uniform(1.2, 4.5) + {1: 0.0, 2: 0.4, 3: 0.8}[device_gen]
        sig = min(5.0, round(sig, 2))
        latency = rng.gauss({1: 180, 2: 110, 3: 65}[device_gen], {1: 45, 2: 25, 3: 12}[device_gen])
        gps_acc = rng.gauss({1: 14, 2: 9, 3: 5}[device_gen], {1: 4, 2: 2.5, 3: 1.2}[device_gen])
        loss = rng.uniform({1: 0.008, 2: 0.004, 3: 0.001}[device_gen], {1: 0.03, 2: 0.015, 3: 0.006}[device_gen])
        # occasional nasty outliers
        if rng.random() < 0.002:
            latency *= rng.uniform(3, 8)
        if rng.random() < 0.002:
            gps_acc *= rng.uniform(2, 4)
        return max(0.0, round(latency, 2)), max(1.0, round(gps_acc, 2)), round(loss, 4), sig

    # ---------------- Record generation ----------------
    records = []
    for _ in range(num_records):
        vehicle_id = rng.choice(list(vehicles.keys()))
        v = vehicles[vehicle_id]

        driver_id = rng.choice(list(drivers.keys()))
        d = drivers[driver_id]

        timestamp = fake.date_time_between(start_date='-30d', end_date='now')
        hour = timestamp.hour

        weather, road_type, traffic, rush_hour = sample_context(hour)
        speed = sample_speed(road_type, traffic)
        distance_km = sample_distance_km(road_type)
        idle_min = sample_idle_minutes(road_type, traffic)

        # Route optimization treatment (A/B)
        optimized_flag, dist_mult, idle_mult = route_optimization_effect(v['route_opt_access'])
        distance_km *= dist_mult
        idle_min *= idle_mult

        # Fuel model
        rate_l_per_100 = fuel_rate_l_per_100km(v, speed, weather, traffic, v['device_generation'])
        fuel_from_motion = (distance_km * rate_l_per_100) / 100
        fuel_from_idle = idle_min * idle_burn_lpm(v['vehicle_type'])
        fuel_consumed = fuel_from_motion + fuel_from_idle
        # small random noise, guard against negatives
        fuel_consumed = max(0.2, round(fuel_consumed * rng.uniform(0.97, 1.03), 2))

        # Events — correlated probabilities
        # base rates
        p_brake = 0.06 + (0.04 if rush_hour else 0) + (0.06 if traffic == 'high' else 0) + (0.05 if weather in ['rainy', 'snowy', 'foggy'] else 0)
        p_brake = min(0.5, p_brake)

        # harsh accel depends strongly on training & feedback
        p_harsh = 0.08 if d['driver_training'] == 'untrained' else 0.04
        intervention_active = rng.random() < 0.55  # real-time alerts active this trip
        if intervention_active:
            p_harsh *= 0.55
        harsh_accel = rng.random() < p_harsh

        braking_event = rng.random() < p_brake

        # collision is rare but higher with harsh accel, bad weather, high traffic, high speed
        p_collision = 0.003
        if harsh_accel: p_collision *= 2.0
        if weather in ['rainy', 'snowy']: p_collision *= 2.0
        if traffic == 'high': p_collision *= 1.6
        if speed > 120: p_collision *= 1.8
        if d['driver_training'] == 'untrained': p_collision *= 1.5
        collision_alert = rng.random() < p_collision

        # lane change moderately common, more on highway
        p_lane = 0.10 if road_type == 'highway' else 0.07
        lane_change_event = rng.random() < p_lane

        # Device/system issues — older gen worse; predictive maintenance helps
        p_sensor_fault = (0.006 if v['device_generation'] == 1 else 0.003 if v['device_generation'] == 2 else 0.0015)
        if v['maintenance_type'] == 'predictive':
            p_sensor_fault *= 0.7
        sensor_fault_event = rng.random() < p_sensor_fault

        p_net_delay = (0.004 if v['device_generation'] == 1 else 0.002 if v['device_generation'] == 2 else 0.001)
        network_delay_event = rng.random() < p_net_delay

        p_gps_loss_evt = (0.003 if v['device_generation'] == 1 else 0.0018 if v['device_generation'] == 2 else 0.0008)
        gps_loss_event = rng.random() < p_gps_loss_evt

        # Priority event type
        event_type = 'normal'
        for key in ['collision_alert', 'sensor_fault_event', 'gps_loss_event', 'network_delay_event',
                    'harsh_acceleration_event', 'braking_event', 'lane_change_event']:
            pass
        # We'll compute after we set booleans

        # Device metrics
        latency_ms, gps_acc_m, loss_rate, signal_strength = device_metrics(v['device_generation'])

        # Sensor battery with rare >100% outliers (bad calibration)
        sensor_battery = round(rng.uniform(20, 100), 2)
        if rng.random() < 0.003:
            sensor_battery = round(rng.uniform(101, 145), 2)

        # Maintenance / downtime / cost
        # breakdown probability influenced by age, maintenance type, sensor faults, collision
        p_breakdown = 0.002 + 0.0006 * v['vehicle_age_years']
        if v['maintenance_type'] == 'predictive':
            p_breakdown *= 0.6
        if sensor_fault_event:
            p_breakdown *= 2.5
        if collision_alert:
            p_breakdown *= 4.0
        breakdown_event = rng.random() < p_breakdown

        # downtime hours distribution
        downtime_hours = 0.0
        if breakdown_event and not collision_alert:
            # typical mechanical breakdown
            downtime_hours = max(0.3, rng.gauss(3.5 if v['maintenance_type']=='predictive' else 6.5, 2.0))
        if collision_alert:
            downtime_hours += rng.uniform(8, 36)  # collisions are costly
        downtime_hours = round(downtime_hours, 2)

        # time since last maintenance — predictive tends to keep it lower
        if v['maintenance_type'] == 'predictive':
            time_since_maint_days = max(0.5, rng.gauss(22, 10))
        else:
            time_since_maint_days = max(0.5, rng.gauss(45, 18))
        time_since_maint_days = round(time_since_maint_days, 1)

        # maintenance cost — base + downtime-driven costs; predictive smoother, fewer spikes
        base_maint = 15 if v['maintenance_type']=='predictive' else 10  # predictive has routine checks
        cost = base_maint + downtime_hours * rng.uniform(40, 120)
        if collision_alert:
            cost += rng.uniform(500, 3000)
        maintenance_cost_usd = round(cost * rng.uniform(0.95, 1.05), 2)

        # Determine event_type priority now that booleans are known
        events_dict = {
            'collision_alert': collision_alert,
            'sensor_fault_event': sensor_fault_event,
            'gps_loss_event': gps_loss_event,
            'network_delay_event': network_delay_event,
            'harsh_acceleration_event': harsh_accel,
            'braking_event': braking_event,
            'lane_change_event': lane_change_event
        }
        for k in ['collision_alert','sensor_fault_event','gps_loss_event','network_delay_event',
                  'harsh_acceleration_event','braking_event','lane_change_event']:
            if events_dict[k]:
                event_type = k.replace('_event','').replace('_alert','')
                break

        # risk label: safe unless collision/sensor fault/breakdown. Small label noise (1%).
        risk_label = not (collision_alert or sensor_fault_event or breakdown_event)
        if rng.random() < 0.01:
            risk_label = not risk_label

        # GPS coordinates (with rare Nones already handled in function)
        lat, lon = generate_gps_coordinates()

        # occasional missing fuel row (very rare)
        if rng.random() < 0.0015:
            fuel_consumed_record = None
            rate_record = None
        else:
            fuel_consumed_record = fuel_consumed
            rate_record = rate_l_per_100

        records.append({
            # --- Identifiers ---
            'vehicle_id': vehicle_id,
            'driver_id': driver_id,
            'route_id': f"RTE{rng.randint(100, 999)}",
            'timestamp': timestamp,

            # --- Context ---
            'latitude': lat,
            'longitude': lon,
            'hour_of_day': hour,
            'weather': weather,
            'road_type': road_type,
            'traffic_density': traffic,
            'rush_hour': rush_hour,

            # --- Vehicle & device ---
            'vehicle_type': v['vehicle_type'],
            'vehicle_age_years': v['vehicle_age_years'],
            'device_generation': v['device_generation'],
            'device_cost_usd': v['device_cost_usd'],

            # --- Driver ---
            'driver_experience_years': d['driver_experience_years'],
            'driver_training': d['driver_training'],

            # --- Route treatment ---
            'optimized_route_flag': optimized_flag,

            # --- Telemetry / performance ---
            'speed': speed,
            'distance_traveled_km': round(distance_km, 1),
            'idle_time_minutes': round(idle_min, 1),
            'fuel_consumption_liters': fuel_consumed_record,
            'fuel_rate_l_per_100km': rate_record,

            # --- Device metrics ---
            'sensor_battery': sensor_battery,
            'sensor_signal_strength': signal_strength,
            'data_latency_ms': latency_ms,
            'gps_accuracy_meters': gps_acc_m,
            'packet_loss_rate': loss_rate,

            # --- Events ---
            'braking_event': braking_event,
            'collision_alert': collision_alert,
            'lane_change_event': lane_change_event,
            'harsh_acceleration_event': harsh_accel,
            'sensor_fault_event': sensor_fault_event,
            'network_delay_event': network_delay_event,
            'gps_loss_event': gps_loss_event,
            'event_type': event_type,

            # --- Maintenance outcomes ---
            'maintenance_type': v['maintenance_type'],
            'time_since_last_maintenance_days': time_since_maint_days,
            'breakdown_event': breakdown_event,
            'downtime_hours': downtime_hours,
            'maintenance_cost_usd': maintenance_cost_usd,

            # --- Intervention ---
            'intervention_active': intervention_active,

            # --- Label ---
            'risk_label': risk_label,
        })

    return pd.DataFrame(records)


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------

# Add the parent directory of the current file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

def connect_to_postgres(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DB,
    user=POSTGRES_USER,
    password=POSTGRES_PASSWORD
) -> Optional[psycopg2.extensions.connection]:
    """Connect to PostgreSQL."""
    try:
        return psycopg2.connect(host=host, port=port, database=database, user=user, password=password)
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None



def create_table_if_not_exists(conn, table_name='fleet_data'):
    """Create fleet_data table with all columns needed for the 5 analyses."""
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        -- Identifiers
        vehicle_id VARCHAR(20),
        driver_id VARCHAR(20),
        route_id VARCHAR(20),
        timestamp TIMESTAMP,

        -- Context
        latitude FLOAT,
        longitude FLOAT,
        hour_of_day SMALLINT,
        weather VARCHAR(20),
        road_type VARCHAR(20),
        traffic_density VARCHAR(20),
        rush_hour BOOLEAN,

        -- Vehicle & device
        vehicle_type VARCHAR(20),
        vehicle_age_years SMALLINT,
        device_generation SMALLINT,
        device_cost_usd FLOAT,

        -- Driver
        driver_experience_years SMALLINT,
        driver_training VARCHAR(20),

        -- Route treatment
        optimized_route_flag BOOLEAN,

        -- Telemetry / performance
        speed FLOAT,
        distance_traveled_km FLOAT,
        idle_time_minutes FLOAT,
        fuel_consumption_liters FLOAT,
        fuel_rate_l_per_100km FLOAT,

        -- Device metrics
        sensor_battery FLOAT,
        sensor_signal_strength FLOAT,
        data_latency_ms FLOAT,
        gps_accuracy_meters FLOAT,
        packet_loss_rate FLOAT,

        -- Events
        braking_event BOOLEAN,
        collision_alert BOOLEAN,
        lane_change_event BOOLEAN,
        harsh_acceleration_event BOOLEAN,
        sensor_fault_event BOOLEAN,
        network_delay_event BOOLEAN,
        gps_loss_event BOOLEAN,
        event_type VARCHAR(30),

        -- Maintenance
        maintenance_type VARCHAR(20),
        time_since_last_maintenance_days FLOAT,
        breakdown_event BOOLEAN,
        downtime_hours FLOAT,
        maintenance_cost_usd FLOAT,

        -- Intervention
        intervention_active BOOLEAN,

        -- Label (True = safe, False = risky)
        risk_label BOOLEAN
    );
    """)
    conn.commit()
    cursor.close()


def save_to_postgres(df, conn, table_name='fleet_data'):
    """Bulk insert DataFrame into PostgreSQL."""
    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ','.join(list(df.columns))
    query = f"INSERT INTO {table_name}({cols}) VALUES %s"
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples, page_size=1000)
        conn.commit()
        print(f"Inserted {len(df)} records into {table_name}")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()
    finally:
        cursor.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("Generating realistic synthetic fleet data...")
    df = generate_fleet_data(num_records=100000, fleet_size=500, driver_pool=400)
    print("Generated dataset with", len(df), "records")

    # Optional quick sanity checks (comment out if not needed)
    print("Event rates (sample):")
    print(df[['collision_alert','sensor_fault_event','harsh_acceleration_event','braking_event']].mean())
    print("Mean fuel (L):", df['fuel_consumption_liters'].dropna().mean().round(2))
    print("Downtime (hrs) mean:", df['downtime_hours'].mean().round(2))

    conn = connect_to_postgres()
    if conn:
        print("Connected to PostgreSQL.")
        create_table_if_not_exists(conn)
        save_to_postgres(df, conn)
        conn.close()
    else:
        print("Could not connect to PostgreSQL server.")


if __name__ == "__main__":
    main()


# How to Run Everything:
# docker compose up -d data_generator
# docker-compose run --rm data_generator python data_generation/generate_data.py
