from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PROJECTS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.csv"
TAGS_URL = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/tags.csv"
NUM_LABELS = 5
BATCH_SIZE =16
TEST_SIZE = 0.15
DROPOUT = 0.4
DATA_FILENAME = "tripadvisor_hotel_reviews.csv"
#not in use
# import tomli
# with open("config.toml", mode="rb") as fp:
#     toml_config = tomli.load(fp)



# Since we're using the MLflowCallback here with Optuna, we can either allow all our experiments to be stored under the default mlruns directory that MLflow will create or we can configure that location:
import mlflow
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))


#Logging
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import sys
import logging
import logging

# Get root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create handlers

info_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "info.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
info_handler.setLevel(logging.INFO)
error_handler = logging.handlers.RotatingFileHandler(
    filename=Path(LOGS_DIR, "error.log"),
    maxBytes=10485760,  # 1 MB
    backupCount=10,
)
error_handler.setLevel(logging.ERROR)

# Create formatters
minimal_formatter = logging.Formatter(fmt="%(message)s")
detailed_formatter = logging.Formatter(
    fmt="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
)

# Hook it all up

info_handler.setFormatter(fmt=detailed_formatter)
error_handler.setFormatter(fmt=detailed_formatter)

logger.addHandler(hdlr=info_handler)
logger.addHandler(hdlr=error_handler)


#https://madewithml.com/courses/mlops/logging/
