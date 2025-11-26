"""Exploratory Data Analysis helpers for the heart survival dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import NOTEBOOK_EVAL_DIR

plt.style.use("seaborn-v0_8-darkgrid")


def missing_value_summary(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per feature."""
    return df.isna().sum()


def detect_outliers_iqr(df: pd.DataFrame, features: List[str]) -> Dict[str, int]:
    """Detect outliers with the IQR rule and return counts per feature."""
    outlier_counts = {}
    for feat in features:
        series = pd.to_numeric(df[feat], errors="coerce").dropna()
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_counts[feat] = int(((series < lower) | (series > upper)).sum())
    return outlier_counts


def correlation_heatmap(df: pd.DataFrame, save_path: Path) -> None:
    """Plot and save correlation heatmap."""
    corr = df.apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def feature_distributions(df: pd.DataFrame, save_path: Path) -> None:
    """Plot histogram distribution of all features."""
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    n_cols = 4
    n_rows = int(np.ceil(len(df_numeric.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(df_numeric.columns):
        axes[idx].hist(df_numeric[col].dropna(), bins=20, color="#1f77b4", alpha=0.8)
        axes[idx].set_title(f"Distribution: {col}")
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def survival_comparison(df: pd.DataFrame, target_col: str, save_path: Path) -> None:
    """Plot survival vs non-survival counts."""
    df_numeric = df.copy()
    df_numeric[target_col] = pd.to_numeric(df_numeric[target_col], errors="coerce")
    df_numeric["survival_label"] = df_numeric[target_col].apply(
        lambda x: "Survived" if x == 0 else "High Risk"
    )
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_numeric, x="survival_label", color="#66c2a5")
    plt.title("Survival vs High Risk Counts")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def medical_insights(df: pd.DataFrame) -> List[str]:
    """Generate human-readable insights from the dataset."""
    insights = []
    df_num = df.apply(pd.to_numeric, errors="coerce")
    risk_corr = df_num.corr()["target"].sort_values(ascending=False)
    top_risk = risk_corr.drop("target").head(3)
    insights.append(
        f"Top drivers of high risk: {', '.join(f'{feat} (corr={val:.2f})' for feat, val in top_risk.items())}."
    )
    low_risk = risk_corr.drop("target").tail(3)
    insights.append(
        f"Protective indicators: {', '.join(f'{feat} (corr={val:.2f})' for feat, val in low_risk.items())}."
    )
    return insights


def run_eda(df: pd.DataFrame, output_dir: Path = NOTEBOOK_EVAL_DIR) -> Dict[str, Path]:
    """Run the full EDA suite and persist figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "heatmap": output_dir / "eda_correlation_heatmap.png",
        "distributions": output_dir / "eda_feature_distributions.png",
        "survival": output_dir / "eda_survival_counts.png",
    }
    correlation_heatmap(df, figures["heatmap"])
    feature_distributions(df, figures["distributions"])
    survival_comparison(df, "target", figures["survival"])
    return figures


__all__ = [
    "missing_value_summary",
    "detect_outliers_iqr",
    "correlation_heatmap",
    "feature_distributions",
    "survival_comparison",
    "medical_insights",
    "run_eda",
]

