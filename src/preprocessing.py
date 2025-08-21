import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from feature_engineering import engineer_features

PROCESSED_DIR = Path("data/processed")


def create_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split data into train and validation sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str = "Survived"):
    """Preprocess data with improved feature engineering."""
    # Apply feature engineering
    train_df, test_df = engineer_features(train_df, test_df)
    
    # Separate features and target
    y = train_df[target]
    X = train_df.drop(columns=[target])
    X_test = test_df.copy()
    
    # Define features to use (drop non-predictive columns)
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    available_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    
    X = X.drop(columns=available_cols_to_drop)
    X_test = X_test.drop(columns=available_cols_to_drop)
    
    # Handle any remaining missing values
    for col in X.columns:
        if X[col].dtype in ['object']:
            # For any remaining categorical columns, use mode
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
            X[col] = X[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
        else:
            # For numerical columns, use median
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # One-hot encode categorical variables consistently across train and test
    categorical_cols = [
        "Pclass",
        "Sex",
        "Embarked",
        "Title",
        "AgeGroup",
        "FareBin",
        "Deck",
    ]
    combined = pd.concat([X, X_test], axis=0)
    combined = pd.get_dummies(
        combined, columns=[c for c in categorical_cols if c in combined.columns], drop_first=True
    )
    n_train = len(X)
    X = combined.iloc[:n_train, :]
    X_test = combined.iloc[n_train:, :]

    # Convert to numpy arrays
    X_processed = X.values.astype(np.float32)
    X_test_processed = X_test.values.astype(np.float32)

    feature_names = X.columns.tolist()

    return X_processed, X_test_processed, y, feature_names, None


def select_features(X: np.ndarray, y: pd.Series, feature_names, k: int = 15):
    """Select top k features using statistical tests and random forest importance."""
    
    # Use SelectKBest for statistical feature selection
    selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_names)))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    
    return selected_features


def save_preprocessed_data(X_train, X_valid, X_test, y_train, y_valid):
    """Save preprocessed data to files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_valid.npy", X_valid)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_valid.npy", y_valid)


def preprocessing_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Complete preprocessing pipeline."""
    # Preprocess data
    X, X_test, y, feature_names, _ = preprocess_data(train_df, test_df)

    # Create DataFrame for easier handling
    X_df = pd.DataFrame(X, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Split training data
    X_train, X_valid, y_train, y_valid = split_data(X_df, y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test_df)

    # Save processed data
    save_preprocessed_data(
        X_train_scaled, X_valid_scaled, X_test_scaled, y_train.values, y_valid.values
    )

    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Validation set size: {X_valid_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, feature_names, scaler


# Legacy functions for compatibility
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using integer codes."""
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes
    return df_encoded


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns, index=df.index)


if __name__ == "__main__":
    from data_loader import load_data

    train_df, test_df = load_data()
    result = preprocessing_pipeline(train_df, test_df)
    print("Preprocessing completed successfully!")