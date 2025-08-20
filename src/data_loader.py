import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def load_data(data_dir: Path = RAW_DATA_DIR):
    """Load train and test datasets from the raw data directory."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def dataset_info(df: pd.DataFrame) -> dict:
    """Return basic information about a dataframe."""
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
    }
    return info


def numerical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return basic statistics for numerical columns."""
    return df.describe(include="number")


def categorical_value_counts(df: pd.DataFrame) -> dict:
    """Return value counts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return {col: df[col].value_counts(dropna=False).to_dict() for col in cat_cols}


def save_data_summary(train: pd.DataFrame, test: pd.DataFrame,
                      path: Path = PROCESSED_DIR / "data_summary.txt") -> None:
    """Save dataset insights to a text file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("TRAIN DATASET INFO\n")
        f.write(str(dataset_info(train)) + "\n\n")

        f.write("TRAIN NUMERICAL STATS\n")
        f.write(str(numerical_stats(train)) + "\n\n")

        f.write("TRAIN CATEGORICAL VALUE COUNTS\n")
        f.write(str(categorical_value_counts(train)) + "\n\n")

        f.write("TEST DATASET INFO\n")
        f.write(str(dataset_info(test)) + "\n\n")

        f.write("TEST NUMERICAL STATS\n")
        f.write(str(numerical_stats(test)) + "\n\n")

        f.write("TEST CATEGORICAL VALUE COUNTS\n")
        f.write(str(categorical_value_counts(test)) + "\n")


if __name__ == "__main__":
    train_df, test_df = load_data()
    save_data_summary(train_df, test_df)
