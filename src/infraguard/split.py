"""Chronological, purged splitting to avoid future-label leakage across boundaries."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.65
    validation_fraction: float = 0.17
    purge_steps: int = 30


def chronological_split(df: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    if not (0 < config.train_fraction < 1 and 0 < config.validation_fraction < 1):
        raise ValueError("Split fractions must be in (0, 1)")
    if config.train_fraction + config.validation_fraction >= 1:
        raise ValueError("Split fractions leave no test data")
    if config.purge_steps < 0:
        raise ValueError("purge_steps must be nonnegative")
    times = pd.Index(df["timestamp"].unique()).sort_values()
    n = len(times)
    first = int(n * config.train_fraction)
    second = int(n * (config.train_fraction + config.validation_fraction))
    gap = config.purge_steps
    if first - gap <= 0 or first + gap >= second - gap or second + gap >= n:
        raise ValueError("Insufficient timestamps for purged split; increase --steps or reduce --purge-steps")
    groups = {
        "train": times[: first - gap],
        "validation": times[first + gap : second - gap],
        "test": times[second + gap :],
    }
    return {name: df[df["timestamp"].isin(index)].copy() for name, index in groups.items()}
