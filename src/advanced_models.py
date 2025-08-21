"""Lightweight advanced models for Titanic project.

Provides utilities for tuning a LightGBM model and building
simple voting ensembles.  These implementations avoid heavy
third‑party dependencies so they work in constrained
execution environments used by the tests.
"""
from __future__ import annotations

from typing import List, Tuple

import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV


def tune_lightgbm(X, y, n_trials: int = 10) -> lgb.LGBMClassifier:
    """Return a LightGBM model tuned with a small random search.

    Parameters
    ----------
    X, y : array-like
        Training data.
    n_trials : int, default=10
        Number of parameter settings sampled.
    """
    param_dist = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.1, 0.05, 0.01],
        "n_estimators": [50, 100, 200],
        "subsample": [0.8, 1.0],
    }
    model = lgb.LGBMClassifier(random_state=42)
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=min(n_trials, 10),
        cv=3,
        scoring="accuracy",
        random_state=42,
    )
    search.fit(X, y)
    return search.best_estimator_


def create_voting_classifier(models: List[Tuple[str, object]]) -> VotingClassifier:
    """Create a simple hard voting classifier from provided estimators."""
    return VotingClassifier(estimators=models, voting="hard")
