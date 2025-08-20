import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)


def load_model(path: Path):
    return joblib.load(path)


def generate_predictions(model, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a model.

    The function prefers ``predict_proba`` if available, returning the
    probability of the positive class.  Some lightweight test doubles (such as
    ``MagicMock``) technically expose a ``predict_proba`` attribute but do not
    return a NumPy array.  In those cases we fall back to ``predict`` to ensure
    a sensible output.
    """

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        if isinstance(probs, np.ndarray):
            return probs[:, 1] if probs.ndim > 1 else probs
    return model.predict(X)


def create_submission(
    passenger_ids: Iterable[int],
    predictions: Iterable[int | float],
    path: Path | None = None,
) -> pd.DataFrame:
    """Create a submission DataFrame and optionally write it to disk."""
    df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    if path is not None:
        df.to_csv(path, index=False)
    return df


def prediction_intervals(probs: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    lower = np.clip(probs - alpha, 0, 1)
    upper = np.clip(probs + alpha, 0, 1)
    return np.vstack([lower, upper]).T


def generate_submission_file(model_path: Path, test_df: pd.DataFrame) -> Path:
    model = load_model(model_path)
    probs = generate_predictions(model, test_df.drop(columns=["PassengerId"], errors="ignore").values)
    preds = (probs > 0.5).astype(int)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = SUBMISSIONS_DIR / f"submission_{model_path.stem}_{timestamp}.csv"
    create_submission(test_df["PassengerId"], preds, sub_path)
    intervals = prediction_intervals(probs)
    np.save(SUBMISSIONS_DIR / f"intervals_{model_path.stem}_{timestamp}.npy", intervals)
    return sub_path


def generate_multiple_submissions(model_paths: List[Path], test_df: pd.DataFrame) -> List[Path]:
    paths = []
    for mp in model_paths:
        paths.append(generate_submission_file(mp, test_df))
    return paths
