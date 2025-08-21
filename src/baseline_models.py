import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_logistic_regression(X, y, test_size: float = 0.2, random_state: int = 42):
    """Fit a logistic regression model and return the model with validation accuracy."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    return model, score


def train_random_forest(X, y, test_size: float = 0.2, random_state: int = 42):
    """Fit a random forest classifier and return the model with validation accuracy."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    return model, score


def evaluate_model(model, X, y, cv: int = 5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=skf)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    return metrics


def logistic_regression_cv(X, y, cv: int = 5):
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    return evaluate_model(model, X, y, cv)


def random_forest_cv(X, y, cv: int = 5):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return evaluate_model(model, X, y, cv)


def gradient_boosting_cv(X, y, cv: int = 5):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    return evaluate_model(model, X, y, cv)


def compare_models(
    X, y, path: Path = RESULTS_DIR / "model_performance.csv"
) -> dict:
    """Evaluate baseline models and compare their accuracy.

    Parameters
    ----------
    X, y : array-like
        Feature matrix and target vector.
    path : Path, optional
        Location where the comparison table will be saved. The file is written
        in CSV format.

    Returns
    -------
    dict
        Mapping of model name to cross-validated accuracy score.
    """

    metrics = {
        "LogisticRegression": logistic_regression_cv(X, y),
        "RandomForest": random_forest_cv(X, y),
    }

    df = pd.DataFrame(metrics).T
    df.to_csv(path)

    # Return only the accuracy scores for simple comparison
    return {name: m["accuracy"] for name, m in metrics.items()}


if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy")
    X_valid = np.load("data/processed/X_valid.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_valid = np.load("data/processed/y_valid.npy")
    X = np.vstack([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    results = compare_models(X, y)
    print(results)
