import pytest

from infraguard.features import make_features
from infraguard.simulate import SimulationConfig, generate
from infraguard.split import SplitConfig, chronological_split


def test_purged_chronological_split():
    df, _ = make_features(generate(SimulationConfig(devices=4, steps=600)))
    splits = chronological_split(df, SplitConfig(purge_steps=30))
    assert all(len(part) > 0 for part in splits.values())
    assert splits["train"].timestamp.max() < splits["validation"].timestamp.min()
    assert splits["validation"].timestamp.max() < splits["test"].timestamp.min()
    assert (splits["validation"].timestamp.min() - splits["train"].timestamp.max()).total_seconds() > 30 * 60


def test_too_short_splits_raise():
    df, _ = make_features(generate(SimulationConfig(devices=1, steps=100)))
    with pytest.raises(ValueError, match="Insufficient"):
        chronological_split(df, SplitConfig(purge_steps=30))
