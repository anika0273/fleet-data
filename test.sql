--DROP TABLE IF EXISTS fleet_data;
--DROP TABLE IF EXISTS fleet_data_cleaned;
--DROP TABLE IF EXISTS fleet_stream_processed;

-- Step 1: data generation
SELECT * FROM fleet_data; -- 100000 rows

-- Step 2: batch ETL
SELECT * FROM fleet_data_cleaned; -- 99512 rows

SELECT * FROM fleet_stream_processed ORDER BY timestamp DESC;
SELECT * FROM fleet_data ORDER BY timestamp DESC;
SELECT * FROM fleet_data_cleaned ORDER BY timestamp DESC;


-- View recent windows
SELECT *
FROM telemetry_metrics_vehicle_5m
ORDER BY window_end DESC, vehicle_id
--LIMIT 50
;

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




-- after setting up grafana and rerunning the streaming_etl.py and produce.py files
SELECT * FROM fleet_stream_processed ORDER BY timestamp;
SELECT MAX(timestamp) FROM fleet_stream_processed;
--TRUNCATE TABLE fleet_stream_processed;
