from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def weighted_voting(models: List[Tuple[str, object]], weights: List[float]) -> VotingClassifier:
    """Create weighted voting ensemble."""
    ensemble = VotingClassifier(estimators=models, voting="soft", weights=weights)
    return ensemble


def optimize_weights(models: List[Tuple[str, object]], X: np.ndarray, y: np.ndarray) -> List[float]:
    """Simple weight optimization via grid search over a small space."""
    best_score = -np.inf
    best_weights = [1.0] * len(models)
    for w in np.linspace(0.1, 1.0, 10):
        weights = [w if i == 0 else 1 for i in range(len(models))]
        ensemble = weighted_voting(models, weights)
        ensemble.fit(X, y)
        score = accuracy_score(y, ensemble.predict(X))
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights


def blending(models: List[object], X_train: np.ndarray, y_train: np.ndarray, X_holdout: np.ndarray) -> np.ndarray:
    """Blend predictions using a holdout set."""
    preds = []
    for model in models:
        clf = clone(model)
        clf.fit(X_train, y_train)
        preds.append(clf.predict_proba(X_holdout)[:, 1])
    return np.mean(preds, axis=0)


def stacking(base_models: List[Tuple[str, object]], meta_model: object, X: np.ndarray, y: np.ndarray) -> object:
    """Create stacking ensemble with multiple base models."""
    ensemble = VotingClassifier(estimators=base_models, voting="soft")
    meta_model.fit(np.vstack([model.predict_proba(X)[:, 1] for _, model in base_models]).T, y)
    return ensemble, meta_model


def dynamic_ensemble_selection(models: List[object], X_valid: np.ndarray, y_valid: np.ndarray) -> object:
    """Select the best performing model on validation set."""
    best_model = None
    best_score = -np.inf
    for model in models:
        score = accuracy_score(y_valid, model.predict(X_valid))
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def evaluate_ensemble(ensemble: VotingClassifier, models: List[object], X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compare ensemble performance vs individual models."""
    results = {}
    ensemble.fit(X, y)
    results["ensemble"] = accuracy_score(y, ensemble.predict(X))
    for i, model in enumerate(models):
        clf = clone(model)
        clf.fit(X, y)
        results[f"model_{i}"] = accuracy_score(y, clf.predict(X))
    return results
