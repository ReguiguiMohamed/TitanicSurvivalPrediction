import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    model = LogisticRegression(max_iter=1000)
    return evaluate_model(model, X, y, cv)


def random_forest_cv(X, y, cv: int = 5):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return evaluate_model(model, X, y, cv)


def gradient_boosting_cv(X, y, cv: int = 5):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    return evaluate_model(model, X, y, cv)


def compare_models(results: dict, path: Path = RESULTS_DIR / "model_performance.csv"):
    df = pd.DataFrame(results).T
    df.to_csv(path)
    return df


if __name__ == "__main__":
    X_train = np.load("data/processed/X_train.npy")
    X_valid = np.load("data/processed/X_valid.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_valid = np.load("data/processed/y_valid.npy")
    X = np.vstack([X_train, X_valid])
    y = np.concatenate([y_train, y_valid])
    results = {
        "LogisticRegression": logistic_regression_cv(X, y),
        "RandomForest": random_forest_cv(X, y),
        "GradientBoosting": gradient_boosting_cv(X, y),
    }
    compare_models(results)
