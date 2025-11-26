"""Data preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import MODELS_DIR

@dataclass
class PreprocessingArtifacts:
    transformer_path: Path
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class Preprocessor:
    """Encapsulates preprocessing: imputation, scaling, encoding, and splitting."""

    def __init__(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.test_size = test_size
        self.random_state = random_state
        self.transformer: ColumnTransformer | None = None

    def _build_transformer(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )
        return transformer

    def fit_transform(self, df: pd.DataFrame, target_col: str = "target") -> PreprocessingArtifacts:
        df_clean = df.replace({pd.NA: np.nan}).apply(pd.to_numeric, errors="coerce")
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col].astype(float).apply(lambda x: 0 if x == 0 else 1)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        self.transformer = self._build_transformer()
        X_train_processed = self.transformer.fit_transform(X_train)
        X_test_processed = self.transformer.transform(X_test)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        transformer_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(self.transformer, transformer_path)

        return PreprocessingArtifacts(
            transformer_path=transformer_path,
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train.values,
            y_test=y_test.values,
        )

    def transform_new(self, df: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("Transformer is not fitted.")
        return self.transformer.transform(df)


__all__ = ["Preprocessor", "PreprocessingArtifacts"]

