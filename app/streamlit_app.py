"""Streamlit interface for the Healthcare Patient Survival Prediction system."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import FEATURE_COLUMNS  # noqa: E402
from src.predict import predict_survival  # noqa: E402


DEFAULT_PATIENT = {
    "age": 55,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 246,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


st.set_page_config(page_title="Healthcare Patient Survival Prediction (ML)", layout="wide")
st.title("Healthcare Patient Survival Prediction (ML)")
st.write(
    "Upload patient data or enter vitals manually to estimate the probability of adverse cardiac outcomes."
)


def manual_input_form() -> Dict[str, float]:
    st.subheader("Manual Input of Vitals")
    cols = st.columns(3)
    inputs = {}
    for idx, feature in enumerate(FEATURE_COLUMNS):
        col = cols[idx % 3]
        inputs[feature] = col.number_input(
            feature,
            value=float(DEFAULT_PATIENT.get(feature, 0.0)),
            format="%.2f",
            step=0.1,
        )
    return inputs


def risk_gauge(probability: float):
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh([0], [probability], color="#d62728" if probability >= 0.5 else "#2ca02c")
    ax.barh([0], [1], color="#f0f0f0", alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Risk Probability")
    ax.set_title("Risk Gauge")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.text(probability, 0, f"{probability:.2f}", va="center", ha="center", color="white")
    plt.tight_layout()
    return fig


def display_prediction_result(result):
    st.metric("High Risk Probability", f"{result.risk_probability:.2f}")
    st.metric("Survival Probability", f"{result.survival_probability:.2f}")
    st.metric("Prediction Label", result.label)
    st.info(result.risk_text)
    st.pyplot(risk_gauge(result.risk_probability))
    feature_img = PROJECT_ROOT / "notebooks" / "evaluation" / "feature_importance.png"
    if feature_img.exists():
        st.subheader("Medical Explanation (Top Features)")
        st.image(str(feature_img))


st.sidebar.header("Prediction Controls")
uploaded_file = st.sidebar.file_uploader("Upload patient data CSV", type=["csv"])
if st.sidebar.button("Predict from Upload") and uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Batch Predictions")
    outputs = []
    for _, row in df.iterrows():
        data = row.to_dict()
        result = predict_survival(data)
        outputs.append(
            {
                "risk_probability": result.risk_probability,
                "survival_probability": result.survival_probability,
                "label": result.label,
            }
        )
    st.dataframe(outputs)

st.markdown("---")
st.subheader("Single Patient Prediction")
patient_input = manual_input_form()
if st.button("Predict"):
    result = predict_survival(patient_input)
    display_prediction_result(result)

