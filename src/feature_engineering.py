import pandas as pd


def extract_title(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Extract passenger title from name data.

    The original implementation expected a DataFrame with a ``Name`` column and
    returned the DataFrame with an added ``Title`` column.  However, the test
    suite sometimes provides a ``Series`` containing names directly and expects
    a ``Series`` of titles in return.  This function now supports both use
    cases:

    * When passed a ``Series`` it returns a ``Series`` of extracted titles.
    * When passed a ``DataFrame`` it returns a copy of the DataFrame with a
      ``Title`` column added.
    """

    # Determine the series of names depending on the input type
    if isinstance(data, pd.Series):
        names = data
        df = None
    else:
        names = data["Name"]
        df = data.copy()

    titles = names.str.extract(r",\s*([^\.]+)\.\s*")[0]
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    titles = titles.where(titles.isin(common), "Rare")

    if df is None:
        return titles

    df["Title"] = titles
    return df


def family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize and IsAlone features."""
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create age groups."""
    bins = [0, 16, 64, 100]
    labels = ["Child", "Adult", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def fare_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create Fare bins using quartiles."""
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck information from Cabin."""
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values for Age, Embarked, and Fare."""
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
    return df


def engineer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a dataset."""
    df = fill_missing_values(df)
    df = extract_title(df)
    df = family_features(df)
    df = age_groups(df)
    df = fare_bins(df)
    df = extract_deck(df)
    return df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply feature engineering to both train and test datasets."""
    train_processed = engineer_dataset(train_df.copy())
    test_processed = engineer_dataset(test_df.copy())
    return train_processed, test_processed


if __name__ == "__main__":
    from data_loader import load_data

    train_df, test_df = load_data()
    train_fe, test_fe = engineer_features(train_df, test_df)
