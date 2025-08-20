import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

VIS_DIR = Path("visualizations")
PROCESSED_DIR = Path("data/processed")


def plot_survival_rate(df: pd.DataFrame, column: str, filename: str) -> pd.Series:
    """Plot survival rate by a specified column and save the plot."""
    rate = df.groupby(column)["Survived"].mean()
    plt.figure(figsize=(6, 4))
    rate.plot(kind="bar")
    plt.ylabel("Survival Rate")
    plt.title(f"Survival Rate by {column}")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / filename)
    plt.close()
    return rate


def correlation_heatmap(df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> pd.DataFrame:
    """Generate a correlation heatmap for numerical features."""
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / filename)
    plt.close()
    return corr


def distribution_plot(df: pd.DataFrame, column: str, filename: str) -> None:
    """Plot distribution for a numerical column."""
    plt.figure(figsize=(6, 4))
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f"{column} Distribution")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / filename)
    plt.close()


def missing_data_heatmap(df: pd.DataFrame, filename: str = "missing_data_heatmap.png") -> None:
    """Visualize missing data patterns."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Data Heatmap")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / filename)
    plt.close()


def survival_by_family_size(df: pd.DataFrame, filename: str = "survival_by_family_size.png") -> pd.Series:
    """Analyze survival rate by family size."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    rate = df.groupby("FamilySize")["Survived"].mean()
    plt.figure(figsize=(8, 4))
    rate.plot(kind="bar")
    plt.ylabel("Survival Rate")
    plt.title("Survival Rate by Family Size")
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(VIS_DIR / filename)
    plt.close()
    return rate


def generate_eda_report(df: pd.DataFrame, path: Path = PROCESSED_DIR / "eda_report.txt") -> None:
    """Run EDA analyses, save plots, and write a summary report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("Survival Rate by Sex:\n" + plot_survival_rate(df, "Sex", "survival_by_sex.png").to_string())
    lines.append("\nSurvival Rate by Pclass:\n" + plot_survival_rate(df, "Pclass", "survival_by_pclass.png").to_string())
    lines.append("\nSurvival Rate by Embarked:\n" + plot_survival_rate(df, "Embarked", "survival_by_embarked.png").to_string())

    age_bins = [0, 16, 64, 100]
    age_labels = ["Child", "Adult", "Senior"]
    df_age = df.copy()
    df_age["AgeGroup"] = pd.cut(df_age["Age"], bins=age_bins, labels=age_labels)
    lines.append("\nSurvival Rate by Age Group:\n" + plot_survival_rate(df_age, "AgeGroup", "survival_by_age_group.png").to_string())

    correlation_heatmap(df)
    distribution_plot(df, "Age", "age_distribution.png")
    distribution_plot(df, "Fare", "fare_distribution.png")
    missing_data_heatmap(df)
    lines.append("\nSurvival Rate by Family Size:\n" + survival_by_family_size(df).to_string())

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))


if __name__ == "__main__":
    from data_loader import load_data

    train_df, _ = load_data()
    generate_eda_report(train_df)
