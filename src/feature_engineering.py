import pandas as pd


def extract_title(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Extract passenger title from name data."""
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


def create_family_size(df: pd.DataFrame) -> pd.Series:
    """Compute family size from SibSp and Parch columns."""
    return df["SibSp"] + df["Parch"] + 1


def family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize and IsAlone indicators to the dataframe."""
    df = df.copy()
    df["FamilySize"] = create_family_size(df)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create age groups and encode them as integers."""
    df = df.copy()
    bins = [0, 16, 64, 100]
    labels = [0, 1, 2]  # Use integers instead of strings
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels)
    df["AgeGroup"] = df["AgeGroup"].astype(float)  # Handle NaN properly
    return df


def fare_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create Fare bins using quartiles and encode as integers."""
    df = df.copy()
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=[0, 1, 2, 3])
    df["FareBin"] = df["FareBin"].astype(float)  # Handle NaN properly
    return df


def extract_deck(df: pd.DataFrame) -> pd.DataFrame:
    """Extract deck information from Cabin and encode numerically."""
    df = df.copy()
    deck = df["Cabin"].str[0].fillna("U")
    
    # Map deck letters to numbers for better model performance
    deck_mapping = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 
        'F': 6, 'G': 7, 'T': 8, 'U': 0  # U for Unknown
    }
    df["Deck"] = deck.map(deck_mapping)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with more sophisticated methods."""
    df = df.copy()
    
    if "Age" in df.columns:
        # Fill Age based on Title groups for better accuracy
        if "Title" not in df.columns:
            df = extract_title(df)
        
        # use overall median as fallback for title groups with no observed ages
        global_median = df["Age"].median() if not df["Age"].dropna().empty else 30.0
        age_medians = df.groupby("Title")["Age"].median().fillna(global_median)
        df["Age"] = df.apply(
            lambda row: age_medians.get(row["Title"], global_median) if pd.isna(row["Age"]) else row["Age"], 
            axis=1
        )
    
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    if "Fare" in df.columns:
        # Fill Fare based on Pclass for better accuracy
        fare_medians = df.groupby("Pclass")["Fare"].median()
        df["Fare"] = df.apply(
            lambda row: fare_medians[row["Pclass"]] if pd.isna(row["Fare"]) else row["Fare"],
            axis=1
        )
    
    return df


def engineer_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a dataset."""
    df = df.copy()
    df = handle_missing_values(df)
    df = extract_title(df)
    df = family_features(df)
    df = age_groups(df)
    df = fare_bins(df)
    df = extract_deck(df)
    
    # Create additional useful features
    df["TicketLength"] = df["Ticket"].astype(str).str.len()
    df["NameLength"] = df["Name"].astype(str).str.len()
    
    # Encode Sex numerically
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    # Encode Embarked numerically
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    
    # Encode Title numerically
    title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}
    df["Title"] = df["Title"].map(title_mapping)
    
    return df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Apply feature engineering to both train and test datasets."""
    train_processed = engineer_dataset(train_df.copy())
    test_processed = engineer_dataset(test_df.copy())
    return train_processed, test_processed


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper to apply standard feature engineering to a dataset."""
    return engineer_dataset(df.copy())


if __name__ == "__main__":
    from data_loader import load_data
    
    train_df, test_df = load_data()
    train_fe, test_fe = engineer_features(train_df, test_df)
    
    print("Feature engineering completed!")
    print(f"Train shape: {train_fe.shape}")
    print(f"Test shape: {test_fe.shape}")
    print(f"Features: {list(train_fe.columns)}")
    print(f"Missing values in train: {train_fe.isnull().sum().sum()}")
    print(f"Missing values in test: {test_fe.isnull().sum().sum()}")