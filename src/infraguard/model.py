"""Train, evaluate, persist, and use the predictive maintenance models."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from infraguard.features import make_features
from infraguard.split import SplitConfig, chronological_split

TARGET = "failure_next_horizon"


def _probs(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.asarray(model.predict_proba(frame.loc[:, features])[:, 1])


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Select on validation labels only, never test labels."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    pred = (probabilities >= threshold).astype(int)
    y = np.asarray(y_true, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "positive_labels": int(y.sum()),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probabilities)) if y.sum() else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(
    raw: pd.DataFrame,
    output_dir: str | Path,
    *,
    split_config: SplitConfig | None = None,
    seed: int = 42,
    estimators: int = 180,
) -> dict:
    """Train XGBoost and logistic-regression baseline, evaluate on held-out test."""
    if TARGET not in raw:
        raise ValueError(f"Missing target column: {TARGET}; training needs labeled historical data")
    if estimators < 1:
        raise ValueError("estimators must be positive")
    split_config = split_config or SplitConfig()
    prepared, columns = make_features(raw)
    if prepared[TARGET].isna().any() or not prepared[TARGET].isin([0, 1]).all():
        raise ValueError("Target labels must be binary without missing values")
    splits = chronological_split(prepared, split_config)
    train_df, val_df, test_df = (splits[name] for name in ("train", "validation", "test"))
    y_train = train_df[TARGET].astype(int).to_numpy()
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()
    if any(len(np.unique(y)) != 2 for y in (y_train, y_val, y_test)):
        raise ValueError("All splits need failures and non-failures; increase --devices/--steps")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Fit imputation using train rows only, not validation/test rows.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_train = imputer.fit_transform(train_df[columns])
    x_val = imputer.transform(val_df[columns])
    x_test = imputer.transform(test_df[columns])
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    classifier = XGBClassifier(
        n_estimators=estimators,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        scale_pos_weight=negatives / max(positives, 1),
        early_stopping_rounds=20 if estimators >= 30 else None,
    )
    classifier.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    val_probs = classifier.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_probs)
    test_probs = classifier.predict_proba(x_test)[:, 1]
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=800, class_weight="balanced", random_state=seed)),
        ]
    )
    logistic.fit(x_train, y_train)
    baseline_val_probs = logistic.predict_proba(x_val)[:, 1]
    baseline_threshold = select_threshold(y_val, baseline_val_probs)
    baseline_test_probs = logistic.predict_proba(x_test)[:, 1]
    bundle = {
        "imputer": imputer,
        "model": classifier,
        "features": columns,
        "threshold": threshold,
    }
    joblib.dump(bundle, out / "model.joblib")
    joblib.dump({"imputer": imputer, "model": logistic, "features": columns, "threshold": baseline_threshold}, out / "baseline.joblib")
    report = {
        "data_origin": "user_provided_or_synthetic; inspect dataset provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "label": TARGET,
        "split": asdict(split_config),
        "features": columns,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "split_boundaries": {
            key: {"first": str(value["timestamp"].min()), "last": str(value["timestamp"].max())}
            for key, value in splits.items()
        },
        "xgboost": {
            "validation": binary_metrics(y_val, val_probs, threshold),
            "test": binary_metrics(y_test, test_probs, threshold),
            "best_iteration": int(getattr(classifier, "best_iteration", estimators - 1)) if estimators >= 30 else None,
        },
        "logistic_regression": {
            "validation": binary_metrics(y_val, baseline_val_probs, baseline_threshold),
            "test": binary_metrics(y_test, baseline_test_probs, baseline_threshold),
        },
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_bundle(model_dir: str | Path, baseline: bool = False) -> dict:
    file = Path(model_dir) / ("baseline.joblib" if baseline else "model.joblib")
    if not file.is_file():
        raise FileNotFoundError(f"No trained model found at {file}. Run `infraguard train` first.")
    # Only load artifacts from trusted sources; joblib uses pickle under the hood.
    return joblib.load(file)


def predict(raw: pd.DataFrame, model_dir: str | Path, *, baseline: bool = False) -> pd.DataFrame:
    bundle = load_bundle(model_dir, baseline=baseline)
    features_df, _ = make_features(raw)
    features = bundle["features"]
    x = bundle["imputer"].transform(features_df.loc[:, features])
    probs = bundle["model"].predict_proba(x)[:, 1]
    result = features_df[["device_id", "timestamp", "anomaly_score"]].copy()
    result["failure_probability"] = probs
    result["risk_alert"] = (probs >= bundle["threshold"]).astype(int)
    result["decision_threshold"] = bundle["threshold"]
    return result


def evaluate(raw: pd.DataFrame, model_dir: str | Path, baseline: bool = False) -> dict:
    """Re-evaluate the stored model on the chronological held-out test split.

    Use ONLY datasets of the same provenance and timeline as original training.
    """
    report_path = Path(model_dir) / "metrics.json"
    training_report = json.loads(report_path.read_text())
    split = SplitConfig(**training_report["split"])
    features_df, _ = make_features(raw)
    test_df = chronological_split(features_df, split)["test"]
    boundaries = training_report["split_boundaries"]["test"]
    if str(test_df["timestamp"].min()) != boundaries["first"] or str(test_df["timestamp"].max()) != boundaries["last"]:
        raise ValueError("Input does not match original training test-window timestamps")
    bundle = load_bundle(model_dir, baseline=baseline)
    probabilities = bundle["model"].predict_proba(bundle["imputer"].transform(test_df[bundle["features"]]))[:, 1]
    return binary_metrics(test_df[TARGET].to_numpy(), probabilities, bundle["threshold"])
