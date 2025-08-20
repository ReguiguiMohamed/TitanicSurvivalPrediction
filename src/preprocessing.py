import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import engineer_features

PROCESSED_DIR = Path("data/processed")


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def create_preprocessing_pipeline(categorical_features, numerical_features):
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numerical_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "Survived"):
    train_df, test_df = engineer_features(train_df, test_df)
    y = train_df[target]
    X = train_df.drop(columns=[target])
    X_test = test_df.copy()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    X_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    return X_processed, X_test_processed, y, feature_names, preprocessor


def select_features(X: np.ndarray, y: pd.Series, feature_names, top_n: int = 10, corr_threshold: float = 0.9):
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    df_reduced = df.drop(columns=to_drop)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df_reduced, y)
    importances = pd.Series(rf.feature_importances_, index=df_reduced.columns)
    selected = importances.sort_values(ascending=False).head(top_n).index.tolist()
    return selected


def save_preprocessed_data(X_train, X_valid, X_test, y_train, y_valid):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_valid.npy", X_valid)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_valid.npy", y_valid)


def preprocessing_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X, X_test, y, feature_names, preprocessor = preprocess_data(train_df, test_df)
    selected_features = select_features(X, y, feature_names)
    X_df = pd.DataFrame(X, columns=feature_names)[selected_features]
    X_test_df = pd.DataFrame(X_test, columns=feature_names)[selected_features]
    X_train, X_valid, y_train, y_valid = split_data(X_df, y)
    save_preprocessed_data(X_train.values, X_valid.values, X_test_df.values, y_train.values, y_valid.values)
    return X_train, X_valid, X_test_df, y_train, y_valid, selected_features, preprocessor


if __name__ == "__main__":
    from data_loader import load_data

    train_df, test_df = load_data()
    preprocessing_pipeline(train_df, test_df)
