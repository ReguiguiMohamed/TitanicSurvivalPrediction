"""Simple reporting utilities for the Titanic project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_performance_report(metrics: Dict[str, float], path: Path = RESULTS_DIR / "performance_report.txt") -> Path:
    """Save overall performance metrics to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    return path


def save_data_summary(df: pd.DataFrame, path: Path = RESULTS_DIR / "data_summary.csv") -> Path:
    """Persist basic dataset description."""
    summary = df.describe(include="all")
    summary.to_csv(path)
    return path


def save_model_comparison(results: pd.DataFrame, path: Path = RESULTS_DIR / "model_comparison.csv") -> Path:
    """Store model comparison results."""
    results.to_csv(path, index=False)
    return path


def final_submission_report(chosen_model: str, path: Path = RESULTS_DIR / "submission_report.txt") -> Path:
    """Create a small justification report for the selected model."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Selected model: {chosen_model}\n")
        f.write("This model was chosen based on cross-validation accuracy.\n")
    return path


__all__ = [
    "save_performance_report",
    "save_data_summary",
    "save_model_comparison",
    "final_submission_report",
]
