"""Synthetic infrastructure telemetry with prospective failures.

All records are synthetic. The simulator does not reproduce actual enterprise
outage rates or imply a particular real-world model performance.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    devices: int = 200
    steps: int = 5200
    interval_minutes: int = 1
    horizon_steps: int = 30
    seed: int = 42
    start: str = "2025-01-01T00:00:00"

    def validate(self) -> None:
        if self.devices < 1:
            raise ValueError("devices must be >= 1")
        if self.steps < 100:
            raise ValueError("steps must be >= 100")
        if self.interval_minutes < 1:
            raise ValueError("interval_minutes must be >= 1")
        if not 1 <= self.horizon_steps < self.steps:
            raise ValueError("horizon_steps must be between 1 and steps - 1")


def _failures(rng: np.random.Generator, steps: int) -> list[int]:
    """Sample separated failure events (some devices never fail)."""
    if rng.random() < 0.12:
        return []
    events: list[int] = []
    cursor = int(rng.integers(75, 180))
    while cursor < steps - 20:
        events.append(cursor)
        cursor += int(rng.integers(170, 410))
    return events


def generate(config: SimulationConfig) -> pd.DataFrame:
    """Generate a labeled, chronologically ordered multi-device dataset.

    Label is 1 iff the *next* failure occurs in (t, t + horizon_steps].
    failure_event is audit-only; it is never a model input.
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    times = pd.date_range(config.start, periods=config.steps, freq=f"{config.interval_minutes}min")
    t = np.arange(config.steps)
    frames = []
    for i in range(config.devices):
        local = np.random.default_rng(rng.integers(0, 2**32 - 1))
        events = _failures(local, config.steps)
        failures = np.zeros(config.steps, dtype=np.int8)
        degradation = np.zeros(config.steps, dtype=np.float64)
        for event in events:
            failures[event] = 1
            lead = int(local.integers(38, 76))
            lo = max(0, event - lead)
            # Smooth accelerating degradation before a failure, not after it.
            degradation[lo:event] += np.linspace(0.0, 1.0, event - lo) ** 1.5
        degradation = np.clip(degradation, 0, 1.5)
        # Short benign anomalies prevent the classifier from treating every spike as failure.
        benign = np.zeros(config.steps)
        for _ in range(max(1, config.steps // 250)):
            pos = int(local.integers(10, config.steps - 10))
            width = min(int(local.integers(4, 13)), config.steps - pos)
            benign[pos : pos + width] += float(local.uniform(0.2, 0.7))
        phase = local.uniform(-np.pi, np.pi)
        cycle = np.sin(2 * np.pi * t / 1440 + phase)
        cpu = np.clip(45 + 10 * cycle + local.normal(0, 7, config.steps) + 29 * degradation + 16 * benign + local.normal(0, 3), 0, 100)
        memory = np.clip(51 + 3 * cycle + local.normal(0, 5, config.steps) + 28 * degradation + 7 * benign + local.normal(0, 3), 0, 100)
        temp = np.clip(46 + 0.18 * cpu + local.normal(0, 2.2, config.steps) + 15 * degradation + 8 * benign + local.normal(0, 2), 15, 120)
        power = np.clip(180 + 1.1 * cpu + local.normal(0, 9, config.steps) + 50 * degradation + 30 * benign, 20, 500)
        fan = np.clip(1600 + 20 * (temp - 45) + local.normal(0, 125, config.steps), 500, 6000)
        disk = np.clip(45 + 0.42 * cpu + local.normal(0, 12, config.steps) + 20 * benign, 0, 300)
        next_events = np.searchsorted(np.asarray(events, dtype=int), t, side="right")
        next_pos = np.full(config.steps, config.steps + config.horizon_steps, dtype=int)
        if events:
            existing = next_events < len(events)
            next_pos[existing] = np.asarray(events)[next_events[existing]]
        target = ((next_pos - t) <= config.horizon_steps).astype(np.int8)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "device_id": f"server-{i:04d}",
                    "cpu_pct": cpu.round(3),
                    "memory_pct": memory.round(3),
                    "temperature_c": temp.round(3),
                    "power_w": power.round(3),
                    "fan_rpm": fan.round(3),
                    "disk_io_mb_s": disk.round(3),
                    "failure_event": failures,
                    "failure_next_horizon": target,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
