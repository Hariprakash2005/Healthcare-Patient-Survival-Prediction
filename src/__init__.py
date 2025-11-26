"""Core package for the Healthcare Patient Survival Prediction system."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOK_EVAL_DIR = PROJECT_ROOT / "notebooks" / "evaluation"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "MODELS_DIR",
    "NOTEBOOK_EVAL_DIR",
]

