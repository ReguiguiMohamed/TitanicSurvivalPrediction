"""Tools for optimising Kaggle submissions.

This module provides a small collection of helpers that can be used to
calibrate prediction probabilities, search decision thresholds and build
simple ensembles.  While intentionally lightweight, these utilities make
it easy to experiment with different submission strategies.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities into binary predictions given a threshold."""
    return (probs >= threshold).astype(int)


def conservative_strategy(probs: np.ndarray) -> np.ndarray:
    """Conservative prediction with higher threshold (0.6)."""
    return apply_threshold(probs, 0.6)


def aggressive_strategy(probs: np.ndarray) -> np.ndarray:
    """Aggressive prediction with lower threshold (0.4)."""
    return apply_threshold(probs, 0.4)


def calibrate_predictions(model: ClassifierMixin, X: np.ndarray, y: np.ndarray, method: str = "sigmoid") -> ClassifierMixin:
    """Return a probability calibrated version of ``model``."""
    logger.info("Calibrating model using %s method", method)
    calibrator = CalibratedClassifierCV(model, method=method, cv=3)
    calibrator.fit(X, y)
    return calibrator


def optimize_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1") -> float:
    """Search for the threshold that maximises the chosen metric."""
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = apply_threshold(probs, t)
        score = f1_score(y_true, preds) if metric == "f1" else accuracy_score(y_true, preds)
        if score > best_score:
            best_score, best_t = score, t
    logger.info("Best threshold %.2f with score %.3f", best_t, best_score)
    return float(best_t)


def ensemble_predictions(probs_dict: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Combine prediction probabilities from multiple models."""
    models = list(probs_dict.keys())
    if weights is None:
        weights = {m: 1.0 for m in models}
    total = np.zeros_like(next(iter(probs_dict.values())))
    weight_sum = 0.0
    for m in models:
        total += probs_dict[m] * weights.get(m, 1.0)
        weight_sum += weights.get(m, 1.0)
    avg_probs = total / weight_sum
    return apply_threshold(avg_probs, 0.5)


def prediction_confidence(probs: np.ndarray) -> np.ndarray:
    """Return confidence scores where 1 indicates certainty."""
    return 1 - np.abs(probs - 0.5) * 2


def ab_test_split(preds_a: np.ndarray, preds_b: np.ndarray) -> Dict[str, int]:
    """Simple A/B test result counts where predictions differ."""
    diff = preds_a != preds_b
    return {
        "n_total": int(len(preds_a)),
        "n_diff": int(diff.sum()),
    }


__all__ = [
    "apply_threshold",
    "conservative_strategy",
    "aggressive_strategy",
    "calibrate_predictions",
    "optimize_threshold",
    "ensemble_predictions",
    "prediction_confidence",
    "ab_test_split",
]
