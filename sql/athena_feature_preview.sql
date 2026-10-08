-- Example causal SQL features in Athena using preceding rows in each device.
-- Requires dataset published as Parquet using `infraguard aws publish`.
SELECT
    device_id,
    timestamp,
    cpu_pct,
    temperature_c,
    power_w,
    AVG(cpu_pct) OVER (
        PARTITION BY device_id ORDER BY timestamp
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS cpu_mean_15,
    AVG(temperature_c) OVER (
        PARTITION BY device_id ORDER BY timestamp
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS temp_mean_15,
    AVG(power_w) OVER (
        PARTITION BY device_id ORDER BY timestamp
        ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
    ) AS power_mean_15
FROM infraguard.telemetry;
