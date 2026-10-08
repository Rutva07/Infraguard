import numpy as np
import pandas as pd
import pytest

from infraguard.features import make_features
from infraguard.simulate import SimulationConfig, generate


def test_features_use_no_future_data_and_no_labels():
    df = generate(SimulationConfig(devices=3, steps=180))
    original, cols = make_features(df)
    modified = df.copy()
    # Mutating only a future row must never change any earlier feature value.
    modified.loc[modified.device_id == "server-0000", "cpu_pct"] = (
        modified.loc[modified.device_id == "server-0000", "cpu_pct"].to_numpy()
    )
    at_future = modified.index[modified.device_id == "server-0000"][90]
    modified.loc[at_future, "cpu_pct"] = 100
    modified["failure_event"] = 1 - modified["failure_event"]
    modified["failure_next_horizon"] = 1 - modified["failure_next_horizon"]
    changed, changed_cols = make_features(modified)
    assert cols == changed_cols
    assert not any("failure" in name for name in cols)
    before = original[(original.device_id == "server-0000") & (original.timestamp < original.loc[at_future, "timestamp"])]
    after = changed[(changed.device_id == "server-0000") & (changed.timestamp < changed.loc[at_future, "timestamp"])]
    np.testing.assert_allclose(before[cols].to_numpy(), after[cols].to_numpy(), equal_nan=True)


def test_missing_data_and_duplicate_identifiers_rejected():
    with pytest.raises(ValueError, match="Missing telemetry"):
        make_features(pd.DataFrame({"device_id": ["a"]}))
    df = generate(SimulationConfig(devices=1, steps=100))
    with pytest.raises(ValueError, match="Duplicate"):
        make_features(pd.concat([df.iloc[:1], df]))
