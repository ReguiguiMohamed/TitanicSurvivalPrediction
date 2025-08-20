from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if set(["Age", "Pclass"]).issubset(df.columns):
        df["AgePclass"] = df["Age"] * df["Pclass"]
    if set(["Fare", "Pclass"]).issubset(df.columns):
        df["FarePclass"] = df["Fare"] * df["Pclass"]
    return df


def add_polynomial_features(df: pd.DataFrame, numerical_cols: Iterable[str], degree: int = 2) -> pd.DataFrame:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numerical_cols])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols))
    df = df.drop(columns=numerical_cols).reset_index(drop=True)
    df = pd.concat([df, poly_df], axis=1)
    return df


def binning_strategy(df: pd.DataFrame, col: str, bins: int, labels: List[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    df[col + "_bin"] = pd.cut(df[col], bins=bins, labels=labels)
    return df


def rare_category_encoding(df: pd.DataFrame, col: str, threshold: float = 0.01) -> pd.DataFrame:
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    df[col] = df[col].replace(rare, "Rare")
    return df


def feature_clustering(df: pd.DataFrame, cols: Iterable[str], n_clusters: int = 5) -> pd.DataFrame:
    df = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df[list(cols)].fillna(0))
    df["cluster"] = clusters
    return df


def test_feature_combinations(df: pd.DataFrame, y: np.ndarray, feature_sets: Dict[str, List[str]], model) -> Dict[str, float]:
    from sklearn.model_selection import cross_val_score

    scores: Dict[str, float] = {}
    for name, features in feature_sets.items():
        X = df[features]
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        scores[name] = cv_scores.mean()
    return scores
