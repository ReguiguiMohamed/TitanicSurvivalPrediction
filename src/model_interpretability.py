from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

RESULTS_DIR = Path("results/interpretability")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_importance(model, feature_names: Iterable[str], path: Path) -> Path:
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Model lacks feature_importances_ attribute")
    importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), np.array(list(feature_names))[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def shap_analysis(model, X: np.ndarray, path: Path) -> Path:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(path)
    plt.close()
    return path


def plot_partial_dependence(model, X: pd.DataFrame, features: list[str], path: Path) -> Path:
    PartialDependenceDisplay.from_estimator(model, X, features)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def explain_individual(model, X: np.ndarray, index: int, path: Path) -> Path:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.force_plot(explainer.expected_value, shap_values[index], X[index], matplotlib=True, show=False)
    plt.savefig(path)
    plt.close()
    return path


def document_model(model, path: Path, description: str) -> Path:
    path.write_text(description)
    return path
