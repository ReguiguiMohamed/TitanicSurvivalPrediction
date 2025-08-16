# src/data_preprocessing.py
# Titanic preprocessing: missing values, feature engineering, encoding, scaling.
# Ready to import and use in training/evaluation scripts.

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------
# Feature Engineering
# ---------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds Titanic-specific features without touching the target column.
    - FamilySize, IsAlone
    - Title (from Name)
    - TicketPrefix
    - CabinDeck, HasCabin
    - FarePerPerson, AgeTimesClass
    - Casts Pclass to string (treat as categorical)
    Note: Missing values for Age/Embarked are handled later by imputers.
    """

    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Royal",
        "Countess": "Royal",
        "Dona": "Royal",
        "Sir": "Royal",
        "Jonkheer": "Royal",
        "Don": "Royal",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer"
    }

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Safety: ensure expected columns exist
        for col in ["SibSp", "Parch", "Fare", "Age", "Pclass", "Name", "Ticket", "Cabin", "Embarked"]:
            if col not in df.columns:
                df[col] = np.nan

        # Family-based
        sibsp = df["SibSp"].fillna(0)
        parch = df["Parch"].fillna(0)
        df["FamilySize"] = sibsp + parch + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # Title from Name
        titles = df["Name"].astype(str).str.extract(r",\s*([^\.]+)\.")[0].str.strip()
        titles = titles.replace(self.title_map)
        df["Title"] = titles.where(titles.isin(["Mr", "Mrs", "Miss", "Master", "Officer", "Royal"]), "Rare")

        # Ticket prefix
        tk = df["Ticket"].astype(str)
        tk = tk.str.replace(r"[./]", "", regex=True)
        tk = tk.str.replace(r"\d", "", regex=True).str.strip().str.upper()
        df["TicketPrefix"] = tk.replace("", "NONE")

        # Cabin features
        cabin = df["Cabin"].fillna("U").astype(str)
        df["CabinDeck"] = cabin.str[0].str.upper()
        df["HasCabin"] = (df["CabinDeck"] != "U").astype(int)

        # Interactions
        fam_size_safe = df["FamilySize"].replace(0, 1)
        df["FarePerPerson"] = df["Fare"] / fam_size_safe
        df["AgeTimesClass"] = df["Age"] * df["Pclass"]

        # Treat Pclass as categorical
        # (keep original Pclass for AgeTimesClass calculation above)
        df["Pclass"] = df["Pclass"].astype("Int64").astype(str)

        return df


# ---------------------------
# Preprocessor Builder
# ---------------------------
def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Tuple[Pipeline, List[str], List[str]]:
    """
    Returns a Pipeline that:
      - Applies FeatureEngineer
      - Imputes + scales numeric features
      - Imputes + one-hot encodes categorical features
    Also returns the resolved numeric and categorical feature lists.
    """

    # Defaults (will be filtered to those present after FeatureEngineer)
    if numeric_features is None:
        numeric_features = [
            "Age",
            "Fare",
            "SibSp",
            "Parch",
            "FamilySize",
            "IsAlone",
            "FarePerPerson",
            "AgeTimesClass",
            "HasCabin",
        ]
    if categorical_features is None:
        categorical_features = [
            "Sex",
            "Embarked",   # imputed with most_frequent
            "Title",
            "CabinDeck",
            "TicketPrefix",
            "Pclass",     # categorical
        ]

    # Pipelines
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # handles Age/Fare/etc.
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # handles Embarked
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer – columns will be validated after FeatureEngineer runs
    selector = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    preprocessor = Pipeline(steps=[
        ("fe", FeatureEngineer()),
        ("ct", selector)
    ])

    return preprocessor, numeric_features, categorical_features


# ---------------------------
# Fit/Transform Helpers
# ---------------------------
def fit_transform(
    df: pd.DataFrame,
    target_col: str = "Survived",
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], Pipeline, List[str]]:
    """
    Fits the preprocessing pipeline on df and returns:
      X_df, y (or None), fitted preprocessor, feature_names
    """
    preprocessor, num, cat = build_preprocessor(numeric_features, categorical_features)

    # Define X/y
    y = df[target_col].astype(int) if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df

    X_mat = preprocessor.fit_transform(X)
    feature_names = _get_feature_names(preprocessor, num, cat)
    X_df = pd.DataFrame(X_mat, columns=feature_names, index=df.index)
    return X_df, y, preprocessor, feature_names


def transform(
    df: pd.DataFrame,
    preprocessor: Pipeline,
    feature_names: Optional[List[str]] = None,
    target_col: str = "Survived"
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Transforms new data with a fitted preprocessor.
    Returns X_df (with same columns/order as fit) and y if present.
    """
    y = df[target_col].astype(int) if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df

    X_mat = preprocessor.transform(X)
    if feature_names is None:
        # Try to infer names from the pipeline
        feature_names = _infer_feature_names(preprocessor) or [f"f{i}" for i in range(X_mat.shape[1])]
    X_df = pd.DataFrame(X_mat, columns=feature_names, index=df.index)
    return X_df, y


# ---------------------------
# Feature Name Utilities
# ---------------------------
def _get_feature_names(preprocessor: Pipeline, num: List[str], cat: List[str]) -> List[str]:
    """
    Builds final feature names: numeric + onehot-expanded categorical.
    """
    ct: ColumnTransformer = preprocessor.named_steps["ct"]

    # numeric names are 1:1 after impute/scale
    num_names = [n for n in num if n in _safe_ct_input_columns(preprocessor)]

    # categorical names via OneHotEncoder
    ohe: OneHotEncoder = ct.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat))

    return num_names + cat_names


def _safe_ct_input_columns(preprocessor: Pipeline) -> List[str]:
    """
    Columns available to the ColumnTransformer after FeatureEngineer.
    """
    # Run FE on an empty dataframe with same columns? Not safe.
    # Instead, pull columns observed during fit:
    ct: ColumnTransformer = preprocessor.named_steps["ct"]
    # The ColumnTransformer stores transformed column names in feature_names_in_
    # for sklearn >= 1.2. Fallback to provided column spec.
    try:
        return list(ct.feature_names_in_)
    except Exception:
        # Fallback: join provided column specs (num + cat)
        cols = []
        for _, _, cols_sel in ct.transformers_:
            if isinstance(cols_sel, list):
                cols.extend(cols_sel)
        return cols


def _infer_feature_names(preprocessor: Pipeline) -> Optional[List[str]]:
    """
    Best-effort reconstruction of feature names after fit.
    """
    try:
        ct: ColumnTransformer = preprocessor.named_steps["ct"]
        num_cols = []
        cat_cols = []
        for name, trans, cols in ct.transformers_:
            if name == "num":
                if isinstance(cols, list):
                    num_cols = cols
            elif name == "cat":
                if isinstance(cols, list):
                    cat_cols = cols

        ohe: OneHotEncoder = ct.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        return num_cols + cat_names
    except Exception:
        return None


__all__ = [
    "FeatureEngineer",
    "build_preprocessor",
    "fit_transform",
    "transform",
]
