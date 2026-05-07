import pytest 
import yaml

import boto3 
from mops.backend.managers import DatasetManager

with open("config.yaml") as f:
        config = yaml.safe_load(f)

@pytest.fixture
def connect_S3():
    s3_client = boto3.client(
        "s3", endpoint_url=f"http://localhost:9000", 
        aws_access_key_id="minioadmin", 
        aws_secret_access_key="minioadmin"
    )
    return s3_client

@pytest.fixture
def get_managers(connect_S3):
    DatasetManager.setup(
        s3_client=connect_S3,
        bucket_name=config["bucket_names"]["datasets"],
        local_datasets_path=config["datasets_root"],
        names=set(config["base_datasets"]),
        base_datasets=set(config["base_datasets"]),
    )
    yield
