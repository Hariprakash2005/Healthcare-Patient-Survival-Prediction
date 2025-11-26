"""Model evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

from . import NOTEBOOK_EVAL_DIR


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core classification metrics."""
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Path
) -> float:
    fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
    auc_score = metrics.roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return auc_score


def plot_feature_importance(
    model, feature_names: np.ndarray, save_path: Path
) -> None:
    """Handles tree-based, logistic, or xgboost feature importance."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    if importances is None:
        return
    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:15]
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=importances[top_idx],
        y=feature_names[top_idx],
        orient="h",
        color="#1f77b4",
    )
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Path
) -> str:
    """Persist a text classification report for quick inspection."""
    report = metrics.classification_report(y_true, y_pred, digits=3)
    save_path.write_text(report, encoding="utf-8")
    return report


def ensure_eval_dir() -> Path:
    NOTEBOOK_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_EVAL_DIR


__all__ = [
    "classification_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_feature_importance",
    "save_classification_report",
    "ensure_eval_dir",
]

