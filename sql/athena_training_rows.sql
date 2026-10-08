-- Run after `infraguard aws publish`.
-- Change the table name if you configured a non-default database/table.
-- Target labels are synthetic when using the included simulator.
SELECT
    timestamp, device_id, cpu_pct, memory_pct, temperature_c,
    power_w, fan_rpm, disk_io_mb_s,
    failure_event, failure_next_horizon
FROM infraguard.telemetry
ORDER BY device_id, timestamp;
