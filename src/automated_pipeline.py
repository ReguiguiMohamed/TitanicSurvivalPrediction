"""End-to-end automated pipeline from raw data to predictions.

The pipeline is intentionally lightweight so unit tests can execute it
quickly.  It demonstrates configuration driven training, automatic model
selection and basic error handling/logging.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

from . import data_loader, data_preprocessing, performance_monitor
from . import report_generator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONFIG_PATH = Path("config/pipeline_config.json")


MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load pipeline configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(cfg: Dict[str, Any]):
    """Instantiate a model from configuration."""
    model_type = cfg["model"]["type"]
    params = cfg["model"].get("params", {})
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type {model_type}")
    return model_cls(**params)


def compare_models(X: np.ndarray, y: np.ndarray, models: Dict[str, Any]) -> Tuple[str, Any]:
    """Return the model with highest cross-validation accuracy."""
    scores = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        scores[name] = np.mean(cv_scores)
        logger.info("Model %s CV accuracy %.3f", name, scores[name])
    best_name = max(scores, key=scores.get)
    logger.info("Selected model %s", best_name)
    return best_name, models[best_name]


def run_pipeline(config_path: Path = CONFIG_PATH) -> Path:
    """Execute the automated training and prediction pipeline.

    Steps
    -----
    1. Load configuration and raw data
    2. Preprocess data
    3. Train models and select the best
    4. Generate predictions for the test set
    5. Save submission file
    6. Log performance metrics
    """
    cfg = load_config(config_path)
    train_df, test_df = data_loader.load_data()
    X_df, y, preprocessor, _ = data_preprocessing.fit_transform(train_df, target_col=cfg["target"])
    X_test, _ = data_preprocessing.transform(test_df, preprocessor)

    X, y_arr = X_df.values, y.values

    # Build candidates
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    best_name, model = compare_models(X, y_arr, candidates)
    model.fit(X, y_arr)

    # Performance log
    metrics = performance_monitor.compute_metrics(y_arr, model.predict(X))
    performance_monitor.log_metrics(metrics, subset="overall")
    report_generator.save_performance_report(metrics.to_dict())
    report_generator.final_submission_report(best_name)

    preds = model.predict(X_test)
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], cfg["target"]: preds})
    submission_path = Path(cfg["paths"]["submission"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    logger.info("Saved submission to %s", submission_path)
    return submission_path


__all__ = ["load_config", "build_model", "compare_models", "run_pipeline"]


if __name__ == "__main__":  # pragma: no cover
    run_pipeline()
