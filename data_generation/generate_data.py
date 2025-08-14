# data_generation/generate_data.py

# Check if PostgreSQL run: pg_isready -h localhost -p 5433

import random
import pandas as pd
from faker import Faker
import psycopg2
from typing import Optional
import psycopg2.extras as extras
from datetime import timedelta

fake = Faker()

"""
def generate_gps_coordinates():
    \"""Generate random GPS coordinates.\"""
    lat = random.uniform(-90, 90)
    lon = random.uniform(-180, 180)
    return lat, lon
"""

# NYC bounding box example (lat/lon)
LAT_MIN, LAT_MAX = 40.4774, 40.9176
LON_MIN, LON_MAX = -74.2591, -73.7004

def generate_gps_coordinates():
    """Generate random GPS coordinates within a region (e.g., NYC)."""
    lat = random.uniform(LAT_MIN, LAT_MAX)
    lon = random.uniform(LON_MIN, LON_MAX)
    return lat, lon

def generate_fleet_data(num_records=100000, fleet_size=500):
    """
    Generate synthetic fleet telemetry data with realism & controlled noise.
    fleet_size = number of unique vehicles in the dataset.
    """

    data = []

    weather_types = ['sunny', 'rainy', 'foggy', 'snowy', 'cloudy']

    road_types = ['highway', 'city', 'rural']

    traffic_density = ['low', 'medium', 'high']

    # Pre-generate consistent vehicle IDs
    vehicle_ids = [fake.license_plate() for _ in range(fleet_size)]

    for _ in range(num_records):
        vehicle_id = random.choice(vehicle_ids)
        timestamp = fake.date_time_between(start_date='-30d', end_date='now') # Random timestamp within the last 30 days
        
        lat, lon = generate_gps_coordinates()
        
        # Introduce missing GPS 0.5% of the time
        if random.random() < 0.005:
            lat, lon = None, None

        speed = round(random.uniform(0, 130), 2)

        # Outlier: occasional unrealistic speed
        if random.random() < 0.002:
            speed = round(random.uniform(150, 250), 2)

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

        # Weather influence: Rain/fog increases collision/braking
        weather = random.choice(weather_types)
        if weather in ['rainy', 'foggy', 'snowy']:
            p_collision *= 2
            p_braking *= 1.5

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

        road_type = random.choice(road_types)
        traffic = random.choice(traffic_density)

        # Add synthetic sensor metrics (simulating battery level, signal strength)
        sensor_battery = round(random.uniform(20, 100), 2)  # percentage
        if random.random() < 0.003:  # outlier battery > 100
            sensor_battery = round(random.uniform(101, 150), 2)

        sensor_signal_strength = round(random.uniform(1, 5), 2)  # arbitrary scale 1-5

        # Add ground truth label example: safe=1, risky=0 (dummy heuristic)
        # e.g., collisions or sensor faults mark 'risky'
        label_risk = False if (events['collision_alert'] or events['sensor_fault_event']) else True

        # Make some labels contradictory (dirty)
        if random.random() < 0.002:
            label_risk = not label_risk

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
    """Insert DataFrame into PostgreSQL quickly using execute_values."""
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

def main():
    print("Generating realistic synthetic fleet data...")
    df = generate_fleet_data(num_records=100000, fleet_size = 500)  # Adjust count as needed
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

#How to Run Everything
# -> Ensure PostgreSQL is running: pg_isready -h localhost -p 5433
# You should see: localhost:5433 - accepting connections
# Create the table in fleet_db: 1. Open pgAdmin, 2. Connect to fleet_db, 3. Open Query Tool, paste the SQL above, and execute.
# Install dependencies (only once): pip install psycopg2-binary pandas faker
# Run the Python script: python generate_data.py
# Check the inserted data in pgAdmin
# In pgAdmin, run: SELECT COUNT(*) FROM fleet_data;SELECT * FROM fleet_data LIMIT 10;