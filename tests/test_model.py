import json

import numpy as np
import pandas as pd

from infraguard.model import evaluate, load_bundle, predict, train
from infraguard.simulate import SimulationConfig, generate
from infraguard.split import SplitConfig


def test_training_roundtrip_and_test_evaluation(tmp_path):
    raw = generate(SimulationConfig(devices=14, steps=530, seed=101))
    report = train(raw, tmp_path, split_config=SplitConfig(purge_steps=15), estimators=40)
    assert (tmp_path / "model.joblib").is_file()
    assert (tmp_path / "baseline.joblib").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text())["seed"] == 42
    assert report["xgboost"]["test"]["n"] > 0
    assert 0 <= report["xgboost"]["test"]["auroc"] <= 1
    prediction = predict(raw, tmp_path)
    assert len(prediction) == len(raw)
    assert np.isfinite(prediction.failure_probability.to_numpy()).all()
    assert prediction.failure_probability.between(0, 1).all()
    verified = evaluate(raw, tmp_path)
    assert verified["auroc"] == report["xgboost"]["test"]["auroc"]
    assert len(load_bundle(tmp_path)["features"]) > 15
    assert "failure_event" not in load_bundle(tmp_path)["features"]
    assert "failure_next_horizon" not in load_bundle(tmp_path)["features"]
