import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARAMS_DIR = RESULTS_DIR / "hyperparameters"
PARAMS_DIR.mkdir(exist_ok=True)


def optimize_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Optuna optimization for XGBoost."""

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **best_params)
    model.fit(X, y)
    joblib.dump(model, PARAMS_DIR / "xgboost_model.pkl")
    with open(PARAMS_DIR / "xgboost_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    return model, best_params


def optimize_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """Optuna optimization for LightGBM."""

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)
    joblib.dump(model, PARAMS_DIR / "lightgbm_model.pkl")
    with open(PARAMS_DIR / "lightgbm_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    return model, best_params


def grid_search_rf(X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """GridSearchCV for Random Forest."""
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    joblib.dump(best_model, PARAMS_DIR / "rf_model.pkl")
    with open(PARAMS_DIR / "rf_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)
    return best_model, grid.best_params_


def tune_model(model, param_grid: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Tuple[object, Dict[str, Any]]:
    """Automatically tune any sklearn-compatible model."""
    grid = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    return best_model, grid.best_params_
