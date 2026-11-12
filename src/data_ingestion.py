from datetime import datetime, timezone
from pathlib import Path
import boto3
import pandas as pd
import yaml
from src.utils.logger import get_logger

logger = get_logger("DataIngestion")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Ingested dataset is empty. Aborting pipeline.")

    if "entity_id" not in df.columns:
        df.insert(0, "entity_id", range(1, len(df) + 1))

    if "event_timestamp" not in df.columns:
        df["event_timestamp"] = datetime.now(timezone.utc).isoformat()

    return df


def ingest_data(config: dict) -> Path:
    ingestion_cfg = config["data_ingestion"]
    source_type = ingestion_cfg["source_type"]
    raw_dir = Path(ingestion_cfg["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / ingestion_cfg["raw_file_name"]

    logger.info(f"Starting data ingestion from source_type: '{source_type}'")

    if source_type == "url":
        df = pd.read_csv(ingestion_cfg["source_url"])
    elif source_type == "s3":
        s3 = boto3.client("s3")
        obj = s3.get_object(
            Bucket=ingestion_cfg["s3_bucket"],
            Key=ingestion_cfg["s3_key"],
        )
        df = pd.read_csv(obj["Body"])
    elif source_type == "local":
        df = pd.read_csv(ingestion_cfg["source_url"])
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    df = validate_and_prepare(df)
    df.to_csv(output_path, index=False)

    logger.info(
        f"Successfully ingested {df.shape[0]} rows and {df.shape[1]} columns to '{output_path}'"
    )
    return output_path


if __name__ == "__main__":
    cfg = load_config()
    ingest_data(cfg)