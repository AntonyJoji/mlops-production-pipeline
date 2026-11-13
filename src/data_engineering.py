from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("DataEngineering")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"entity_id", "event_timestamp", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Drop high-cardinality/sparse columns not needed for baseline modeling
    drop_cols = [c for c in ["Name", "Ticket", "Cabin"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # Impute missing values deterministically
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Ensure clean data types
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

    if df.isnull().sum().sum() > 0:
        raise ValueError("Null values remain after preprocessing.")

    return df


def run_data_engineering(config: dict) -> tuple[Path, Path]:
    raw_path = Path(config["data_ingestion"]["raw_data_dir"]) / config["data_ingestion"]["raw_file_name"]
    eng_cfg = config["data_engineering"]
    processed_dir = Path(eng_cfg["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading raw dataset from '{raw_path}'")
    df = pd.read_csv(raw_path)

    df_clean = clean_and_validate(df)

    train_df, test_df = train_test_split(
        df_clean,
        test_size=eng_cfg["test_size"],
        random_state=eng_cfg["random_state"],
        stratify=df_clean["Survived"],
    )

    train_path = processed_dir / eng_cfg["train_file_name"]
    test_path = processed_dir / eng_cfg["test_file_name"]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train split ({train_df.shape}) to '{train_path}'")
    logger.info(f"Saved test split ({test_df.shape}) to '{test_path}'")
    return train_path, test_path


if __name__ == "__main__":
    cfg = load_config()
    run_data_engineering(cfg)