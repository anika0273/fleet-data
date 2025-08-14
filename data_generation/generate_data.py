# data_generation/generate_data.py

import random
import pandas as pd
from faker import Faker
import psycopg2
from typing import Optional

fake = Faker()

def generate_gps_coordinates():
    """Generate random GPS coordinates."""
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon

def generate_fleet_data(num_records=100000):
    """
    Generate rich synthetic fleet data simulating diverse real-world scenarios.
    
    Args:
        num_records (int): Number of records to generate.

    Returns:
        pd.DataFrame: Generated fleet data with contextual metadata, edge cases, and event flags.
    """

    data = []

    # Define possible categories and event probabilities
    event_types = [
        'normal', 'braking', 'collision', 'lane_change',
        'harsh_acceleration', 'sensor_fault', 'network_delay', 'gps_loss'
    ]

    weather_types = ['sunny', 'rainy', 'foggy', 'snowy', 'cloudy']

    road_types = ['highway', 'city', 'rural']

    traffic_density = ['low', 'medium', 'high']

    for _ in range(num_records):
        vehicle_id = fake.license_plate()
        timestamp = fake.date_time_between(start_date='-1y', end_date='now')
        lat, lon = generate_gps_coordinates()
        speed = round(random.uniform(0, 130), 2)

        hour = timestamp.hour
        rush_hour = (7 <= hour <= 9) or (16 <= hour <= 18)

        # Probabilities influenced by time of day or randomness
        p_braking = 0.15 if rush_hour else 0.07
        p_collision = 0.005
        p_lane_change = 0.1
        p_harsh_accel = 0.03
        p_sensor_fault = 0.002
        p_network_delay = 0.001
        p_gps_loss = 0.001

        # Simulate multi-event occurrences
        events = {
            'braking_event': random.random() < p_braking,
            'collision_alert': random.random() < p_collision,
            'lane_change_event': random.random() < p_lane_change,
            'harsh_acceleration_event': random.random() < p_harsh_accel,
            'sensor_fault_event': random.random() < p_sensor_fault,
            'network_delay_event': random.random() < p_network_delay,
            'gps_loss_event': random.random() < p_gps_loss,
        }

        # Determine a single highest priority event_type for reference
        event_priority = [
            'collision_alert', 'sensor_fault_event', 'gps_loss_event',
            'network_delay_event', 'harsh_acceleration_event', 'braking_event', 'lane_change_event'
        ]
        event_type = 'normal'
        for evt_key in event_priority:
            if events[evt_key]:
                event_type = evt_key.replace('_event','').replace('_alert','')
                break

        weather = random.choice(weather_types)
        road_type = random.choice(road_types)
        traffic = random.choice(traffic_density)

        # Add synthetic sensor metrics (simulating battery level, signal strength)
        sensor_battery = round(random.uniform(20, 100), 2)  # percentage
        sensor_signal_strength = round(random.uniform(1, 5), 2)  # arbitrary scale 1-5

        # Add ground truth label example: safe=1, risky=0 (dummy heuristic)
        # e.g., collisions or sensor faults mark 'risky'
        label_risk = 0 if (events['collision_alert'] or events['sensor_fault_event']) else 1

        data.append({
            'vehicle_id': vehicle_id,
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'speed': speed,
            'event_type': event_type,
            **events,
            'weather': weather,
            'road_type': road_type,
            'traffic_density': traffic,
            'hour_of_day': hour,
            'sensor_battery': sensor_battery,
            'sensor_signal_strength': sensor_signal_strength,
            'risk_label': label_risk
        })

    return pd.DataFrame(data)

def connect_to_postgres(
    host="localhost",
    port="5433",
    database="fleet_db",
    user="postgres",
    password="1234"
) -> Optional[psycopg2.extensions.connection]:
    """Connect to PostgreSQL database."""

    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def create_table_if_not_exists(conn, table_name='fleet_data'):
    """Create the fleet_data table with all columns if not exists."""

    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
    """)
    conn.commit()
    cursor.close()

def save_to_postgres(df, conn, table_name='fleet_data'):
    """Insert all rows from DataFrame into PostgreSQL."""

    cursor = conn.cursor()
    insert_sql = f"""
    INSERT INTO {table_name} (
        vehicle_id, timestamp, latitude, longitude, speed, event_type,
        braking_event, collision_alert, lane_change_event, harsh_acceleration_event,
        sensor_fault_event, network_delay_event, gps_loss_event, weather, road_type,
        traffic_density, hour_of_day, sensor_battery, sensor_signal_strength, risk_label
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    """

    for _, row in df.iterrows():
        cursor.execute(insert_sql, (
            row.vehicle_id, row.timestamp, row.latitude, row.longitude, row.speed, row.event_type,
            row.braking_event, row.collision_alert, row.lane_change_event, row.harsh_acceleration_event,
            row.sensor_fault_event, row.network_delay_event, row.gps_loss_event, row.weather, row.road_type,
            row.traffic_density, row.hour_of_day, row.sensor_battery, row.sensor_signal_strength, row.risk_label
        ))

    conn.commit()
    cursor.close()

def main():
    print("Generating rich synthetic fleet data...")
    df = generate_fleet_data(num_records=100000)  # Adjust count as needed
    print("Generated dataset with", len(df), "records")

    conn = connect_to_postgres()
    if conn:
        print("Connected to PostgreSQL.")
        create_table_if_not_exists(conn)
        print("Created table or confirmed it exists.")
        save_to_postgres(df, conn)
        print("Data insertion completed successfully.")
        conn.close()
    else:
        print("Could not connect to PostgreSQL server.")

if __name__ == "__main__":
    main()
# This script generates synthetic fleet data and saves it to a PostgreSQL database.
# It includes rich contextual metadata, simulates edge cases, and flags events.
# The data can be used for testing, training, or analysis in fleet management systems.
# Ensure PostgreSQL is running and the connection parameters are correct.
# Adjust the number of records as needed for your use case.
# The script uses Faker for realistic data generation and psycopg2 for database interaction.
# Make sure to install the required packages: faker, pandas, psycopg2.
# You can run this script directly to generate and store the data.
# Ensure you have the necessary permissions to create tables and insert data in the PostgreSQL database.
