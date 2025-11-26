"""Inference utilities for serving survival risk predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import joblib
import pandas as pd

from . import MODELS_DIR
from .data_loader import FEATURE_COLUMNS


@dataclass
class PredictionOutput:
    risk_probability: float
    survival_probability: float
    label: str
    risk_text: str


def _load_artifacts():
    model = joblib.load(MODELS_DIR / "model.pkl")
    transformer = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, transformer


def prepare_input(patient_data: Dict[str, float]) -> pd.DataFrame:
    """Order patient data in the exact schema used during training."""
    missing = [col for col in FEATURE_COLUMNS if col not in patient_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    ordered = {col: patient_data[col] for col in FEATURE_COLUMNS}
    return pd.DataFrame([ordered])


def predict_survival(patient_data: Dict[str, float]) -> PredictionOutput:
    model, transformer = _load_artifacts()
    input_df = prepare_input(patient_data)
    transformed = transformer.transform(input_df)
    risk_probability = float(model.predict_proba(transformed)[0, 1])
    survival_probability = 1.0 - risk_probability
    label = "High Risk" if risk_probability >= 0.5 else "Low Risk"
    risk_text = (
        "Elevated probability of adverse outcomes"
        if label == "High Risk"
        else "Likely to have favorable survival outlook"
    )
    return PredictionOutput(
        risk_probability=risk_probability,
        survival_probability=survival_probability,
        label=label,
        risk_text=risk_text,
    )


__all__ = ["predict_survival", "PredictionOutput"]

