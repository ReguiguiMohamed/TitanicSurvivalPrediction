"""Minimal hyperparameter tuning utilities used in the tests.

The real project might use more sophisticated optimisation with Optuna,
XGBoost, etc., but for the purposes of the exercises we only require a
small subset that depends solely on scikit-learn and LightGBM.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def optimize_lightgbm(X, y, n_trials: int = 10) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Tune a LightGBM model using a small random search."""
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
    return search.best_estimator_, search.best_params_


def grid_search_rf(X, y) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Simple grid search for a RandomForest classifier."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
    }
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_
