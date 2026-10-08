import numpy as np
import pandas as pd
import pytest

from infraguard.simulate import SimulationConfig, generate


def test_reproducible_simulation_and_future_label():
    config = SimulationConfig(devices=4, steps=400, horizon_steps=20, seed=42)
    df = generate(config)
    assert len(df) == 1600
    pd.testing.assert_frame_equal(df, generate(config))
    for _, group in df.groupby("device_id"):
        events = np.flatnonzero(group.failure_event.to_numpy())
        target = group.failure_next_horizon.to_numpy()
        for idx in range(len(group)):
            expected = int(np.any((events > idx) & (events <= idx + 20)))
            assert target[idx] == expected


def test_simulation_requires_valid_parameters():
    with pytest.raises(ValueError):
        generate(SimulationConfig(steps=10))
