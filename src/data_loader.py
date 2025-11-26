"""Utilities for loading the heart disease survival dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import RAW_DATA_PATH

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

FEATURE_COLUMNS = COLUMN_NAMES[:-1]


def load_data(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the heart dataset, applying column names and basic cleaning."""
    path = Path(csv_path) if csv_path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    df.replace("?", pd.NA, inplace=True)
    df["target"] = df["target"].astype("float")
    return df


__all__ = ["load_data", "COLUMN_NAMES", "FEATURE_COLUMNS"]

