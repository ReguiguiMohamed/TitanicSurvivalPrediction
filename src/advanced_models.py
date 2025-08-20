import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def tune_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> lgb.LGBMClassifier:
    """Tune LightGBM hyperparameters using Optuna and return best model."""

    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
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
    joblib.dump(model, MODELS_DIR / "lightgbm_optuna.pkl")
    with open(MODELS_DIR / "lightgbm_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    return model


def tune_svm(X: np.ndarray, y: np.ndarray) -> SVC:
    """Tune Support Vector Machine with grid search."""
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    svm = SVC(probability=True)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    joblib.dump(best_model, MODELS_DIR / "svm_grid.pkl")
    with open(MODELS_DIR / "svm_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)
    return best_model


def tune_mlp(X: np.ndarray, y: np.ndarray) -> MLPClassifier:
    """Tune Neural Network (MLPClassifier) parameters."""
    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-4, 1e-3, 1e-2],
    }
    mlp = MLPClassifier(max_iter=1000)
    grid = GridSearchCV(mlp, param_grid, cv=5, scoring="accuracy")
    grid.fit(X, y)
    best_model = grid.best_estimator_
    joblib.dump(best_model, MODELS_DIR / "mlp_grid.pkl")
    with open(MODELS_DIR / "mlp_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)
    return best_model


def create_voting_classifier(models: List[Tuple[str, object]], weights: List[float] | None = None) -> VotingClassifier:
    """Create an ensemble voting classifier from provided models."""
    voting = VotingClassifier(estimators=models, voting="soft", weights=weights)
    return voting


def create_stacking_classifier(base_models: List[Tuple[str, object]], meta_model: object) -> StackingClassifier:
    """Create a stacking ensemble with a meta-learner."""
    stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, passthrough=False)
    return stack


def train_and_save_models(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Train models, create ensembles, and save the best ones."""
    scores: Dict[str, float] = {}

    lgb_model = tune_lightgbm(X, y)
    scores["lightgbm"] = accuracy_score(y, lgb_model.predict(X))

    svm_model = tune_svm(X, y)
    scores["svm"] = accuracy_score(y, svm_model.predict(X))

    mlp_model = tune_mlp(X, y)
    scores["mlp"] = accuracy_score(y, mlp_model.predict(X))

    voting = create_voting_classifier(
        [
            ("lgb", lgb_model),
            ("svm", svm_model),
            ("mlp", mlp_model),
        ]
    )
    voting.fit(X, y)
    joblib.dump(voting, MODELS_DIR / "voting.pkl")
    scores["voting"] = accuracy_score(y, voting.predict(X))

    stacking = create_stacking_classifier(
        [
            ("lgb", lgb_model),
            ("svm", svm_model),
            ("mlp", mlp_model),
        ],
        RandomForestClassifier(n_estimators=100, random_state=42),
    )
    stacking.fit(X, y)
    joblib.dump(stacking, MODELS_DIR / "stacking.pkl")
    scores["stacking"] = accuracy_score(y, stacking.predict(X))

    return scores
