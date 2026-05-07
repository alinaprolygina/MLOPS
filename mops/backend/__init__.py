import boto3
import yaml
from pathlib import Path
from loguru import logger
from mops.backend.managers import DatasetManager, ModelManager


def setup(local: bool, clearml: bool):
    # config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # logs
    LOG_FILE = config["logs"]["backend"]
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=LOG_FILE,
        mode="w",
        rotation="500 MB",
        filter=lambda record: record["extra"].get("logger_name") == "backend",
    )

    # s3_client
    host = "localhost" if local else "host.docker.internal"
    S3_CLEINT = boto3.client(
        "s3", endpoint_url=f"http://{host}:9000", aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin"
    )

    # datasets
    DatasetManager.setup(
        s3_client=S3_CLEINT,
        bucket_name=config["bucket_names"]["datasets"],
        local_datasets_path=config["datasets_root"],
        names=set(config["base_datasets"]),
        base_datasets=set(config["base_datasets"]),
    )

    # models
    ModelManager.setup(
        s3_client=S3_CLEINT,
        clearml=clearml,
        bucket_name=config["bucket_names"]["models"],
    )
