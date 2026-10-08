"""Causal feature engineering from current and historical telemetry only."""

import numpy as np
import pandas as pd

RAW_SIGNALS = [
    "cpu_pct",
    "memory_pct",
    "temperature_c",
    "power_w",
    "fan_rpm",
    "disk_io_mb_s",
]
REQUIRED = ["timestamp", "device_id", *RAW_SIGNALS]
WINDOWS = (5, 15, 30)


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Sort each device's measurements and add trailing-only rolling statistics.

    Do not ever derive features from `failure_event` or `failure_next_horizon`.
    For production streaming, provide the appropriate past-window context.
    """
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing telemetry columns: {missing}")
    if frame.empty:
        raise ValueError("Telemetry data is empty")
    df = frame.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise", utc=True)
    if df.duplicated(["device_id", "timestamp"]).any():
        raise ValueError("Duplicate (device_id, timestamp) measurements")
    df = df.sort_values(["device_id", "timestamp"], kind="stable").reset_index(drop=True)
    for col in RAW_SIGNALS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    group = df.groupby("device_id", sort=False)
    feature_cols = list(RAW_SIGNALS)
    for col in ["cpu_pct", "memory_pct", "temperature_c", "power_w"]:
        # Includes current reading and ONLY previous readings of the same device.
        for window in WINDOWS:
            name = f"{col}_mean_{window}"
            df[name] = group[col].transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
            feature_cols.append(name)
        delta = f"{col}_delta_1"
        df[delta] = group[col].diff().fillna(0)
        feature_cols.append(delta)
        change = f"{col}_minus_mean_15"
        df[change] = df[col] - df[f"{col}_mean_15"]
        feature_cols.append(change)
    df["cpu_temp_interaction"] = df["cpu_pct"] * df["temperature_c"] / 100
    feature_cols.append("cpu_temp_interaction")
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feature_cols.extend(["hour_sin", "hour_cos"])
    # A transparent unsupervised anomaly score; no target information is accessed.
    df["anomaly_score"] = (
        (df["cpu_pct_minus_mean_15"].abs() / 18)
        + (df["temperature_c_minus_mean_15"].abs() / 8)
        + (df["power_w_minus_mean_15"].abs() / 40)
    ) / 3
    feature_cols.append("anomaly_score")
    return df, feature_cols
