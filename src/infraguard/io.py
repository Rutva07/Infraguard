"""File IO used by local pipelines."""

from pathlib import Path

import pandas as pd


def read_frame(path: str | Path) -> pd.DataFrame:
    """Load a .csv or .csv.gz file; parse timestamps during feature engineering."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No input file at {path}")
    if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
        raise ValueError("Input must be .csv or .csv.gz")
    return pd.read_csv(path)


def write_frame(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
        raise ValueError("Output must be .csv or .csv.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
