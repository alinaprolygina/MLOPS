import yaml
from pathlib import Path
from loguru import logger


def setup():
    # config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # logs
    LOG_FILE = config["logs"]["frontend"]
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=LOG_FILE,
        mode="w",
        enqueue=True,
        rotation="500 MB",
        filter=lambda record: record["extra"].get("logger_name") == "frontend",
    )
