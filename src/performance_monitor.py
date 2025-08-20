"""Utility functions for monitoring model performance.

This module tracks model metrics across passenger subsets, logs metrics
and detects performance degradation over time.  Functions are written to
be lightweight and operate on arrays so they can be reused in
experiments and the automated pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = RESULTS_DIR / "performance_log.csv"


@dataclass
class Metrics:
    """Container for common classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    """Return basic classification metrics."""
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )


def subset_performance(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_cols: Iterable[str],
) -> pd.DataFrame:
    """Compute metrics for each subgroup defined by ``group_cols``.

    Parameters
    ----------
    df:
        DataFrame containing passenger features for grouping.
    y_true:
        True labels.
    y_pred:
        Predicted labels.
    group_cols:
        Columns used to split the data.
    """
    data = df.copy()
    data["y_true"] = y_true
    data["y_pred"] = y_pred
    records = []
    for values, group in data.groupby(list(group_cols)):
        metrics = compute_metrics(group["y_true"], group["y_pred"]).to_dict()
        if not isinstance(values, tuple):
            values = (values,)
        record = {col: val for col, val in zip(group_cols, values)}
        record.update(metrics)
        records.append(record)
    return pd.DataFrame(records)


def log_metrics(metrics: Metrics, subset: Optional[str] = None, log_path: Path = LOG_PATH) -> None:
    """Append metrics with timestamp to a CSV log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.utcnow().isoformat(), **metrics.to_dict()}
    if subset:
        row["subset"] = subset
    df = pd.DataFrame([row])
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)
    logger.info("Logged performance metrics%s", f" for {subset}" if subset else "")


def detect_degradation(
    log_path: Path = LOG_PATH,
    metric: str = "accuracy",
    window: int = 3,
    threshold: float = 0.05,
) -> bool:
    """Return ``True`` if the latest metric drops more than ``threshold`` relative
    to the mean of the previous ``window`` entries."""
    if not log_path.exists():
        logger.warning("No performance log found at %s", log_path)
        return False
    df = pd.read_csv(log_path)
    if len(df) <= window:
        return False
    recent = df[metric].iloc[-window - 1 :]
    baseline = recent.iloc[:-1].mean()
    latest = recent.iloc[-1]
    drop = baseline - latest
    return drop > threshold


def validate_model(model, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, group_cols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Generate overall and subset performance metrics for ``model``.

    The function fits the model, computes metrics on the provided data and
    returns a dictionary with overall metrics and subset metrics.
    """
    logger.info("Fitting model %s", model.__class__.__name__)
    model.fit(X, y)
    preds = model.predict(X)
    overall = compute_metrics(y, preds)
    log_metrics(overall)
    subset_df = subset_performance(df, y, preds, group_cols)
    return {"overall": pd.DataFrame([overall.to_dict()]), "subsets": subset_df}


__all__ = [
    "Metrics",
    "compute_metrics",
    "subset_performance",
    "log_metrics",
    "detect_degradation",
    "validate_model",
]
