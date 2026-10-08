-- DuckDB summary against the local `telemetry` relation.
SELECT
    device_id,
    COUNT(*) AS measurements,
    AVG(cpu_pct) AS mean_cpu_pct,
    MAX(temperature_c) AS peak_temperature_c,
    AVG(memory_pct) AS mean_memory_pct,
    AVG(power_w) AS mean_power_w,
    SUM(failure_event) AS observed_failures
FROM telemetry
GROUP BY device_id
ORDER BY observed_failures DESC, device_id;
