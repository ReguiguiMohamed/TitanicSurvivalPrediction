import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def stratified_kfold_cv(model, X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Perform stratified K-fold cross-validation and return scores."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores


def learning_curve_analysis(model, X: np.ndarray, y: np.ndarray, path: Path | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate learning curve data and optionally save a plot."""
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    if path:
        plt.figure()
        plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
        plt.plot(train_sizes, test_scores.mean(axis=1), label="validation")
        plt.legend()
        plt.savefig(path)
        plt.close()
    return train_sizes, train_scores, test_scores


def feature_importance(model, feature_names: Iterable[str]) -> pd.DataFrame:
    """Return feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
    raise AttributeError("Model does not support feature_importances_ attribute")


def model_stability(model_class, params: Dict, X: np.ndarray, y: np.ndarray, seeds: List[int]) -> Dict[int, float]:
    """Test model stability over multiple random seeds."""
    scores: Dict[int, float] = {}
    for seed in seeds:
        params_with_seed = params | {"random_state": seed}
        model = model_class(**params_with_seed)
        cv_scores = stratified_kfold_cv(model, X, y)
        scores[seed] = cv_scores.mean()
    return scores


def validation_metrics_table(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Create a comparison table of validation metrics."""
    data = {name: [scores.mean(), scores.std()] for name, scores in results.items()}
    df = pd.DataFrame(data, index=["mean", "std"]).T
    return df


def select_final_model(results: Dict[str, np.ndarray]) -> str:
    """Select final model based on average CV score and stability."""
    best_model = max(results.items(), key=lambda x: x[1].mean())[0]
    return best_model
