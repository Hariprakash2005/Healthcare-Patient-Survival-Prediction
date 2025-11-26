"""Model training pipeline for patient survival prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import MODELS_DIR
from .data_loader import load_data
from .evaluate import (
    classification_metrics,
    ensure_eval_dir,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    save_classification_report,
)
from .eda import (
    detect_outliers_iqr,
    medical_insights,
    missing_value_summary,
    run_eda,
)
from .preprocessing import Preprocessor

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


def get_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, class_weight="balanced"
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def train_and_select_model() -> Tuple[str, object, Dict[str, Dict[str, float]]]:
    df = load_data()
    eval_dir = ensure_eval_dir()
    run_eda(df, output_dir=eval_dir)
    categorical = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    numeric = [col for col in df.columns if col not in categorical + ["target"]]
    preprocessor = Preprocessor(categorical_features=categorical, numeric_features=numeric)
    artifacts = preprocessor.fit_transform(df)

    X_train, X_test = artifacts.X_train, artifacts.X_test
    y_train, y_test = artifacts.y_train, artifacts.y_test
    transformer = preprocessor.transformer
    feature_names = np.array(transformer.get_feature_names_out())

    models = get_models()
    eval_summaries: Dict[str, Dict[str, float]] = {}
    best_name, best_model, best_auc = "", None, -np.inf
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        metrics_dict = classification_metrics(y_test, y_pred)
        roc_auc = plot_roc_curve(
            y_test,
            y_proba,
            ensure_eval_dir() / f"{name.lower().replace(' ', '_')}_roc.png",
        )
        metrics_dict["roc_auc"] = roc_auc
        eval_summaries[name] = metrics_dict
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model = model
            best_name = name

    assert best_model is not None
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    joblib.dump(feature_names, MODELS_DIR / "feature_names.npy")
    plot_confusion_matrix(
        y_test, best_model.predict(X_test), eval_dir / "confusion_matrix.png"
    )
    plot_feature_importance(
        best_model, feature_names, eval_dir / "feature_importance.png"
    )
    save_classification_report(
        y_test, best_model.predict(X_test), eval_dir / "classification_report.txt"
    )

    eda_summary = {
        "missing_values": missing_value_summary(df).to_dict(),
        "outliers_iqr": detect_outliers_iqr(df, features=numeric),
        "medical_insights": medical_insights(df),
    }

    with open(eval_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_summaries, f, indent=4)
    with open(eval_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=4)

    return best_name, best_model, eval_summaries


if __name__ == "__main__":
    model_name, _, metrics_summary = train_and_select_model()
    print(f"Best model: {model_name}")
    print(json.dumps(metrics_summary[model_name], indent=2))

