# [HSE2024] MLOps Project

## Overview

This project provides a machine learning dashboard where users can interact with various datasets and models, as well as an API that processes model training and predictions. The dashboard and API run independently, with the dashboard interacting with the API through RESTful services. The dashboard is built using **Streamlit**, and the API is developed both with **FastAPI** and **GRPC**.

## Authors
- Vladislav Bizin ([vkbizin@edu.hse.ru](mailto:vkbizin@edu.hse.ru))
- Alexandr Atlasov ([aaatlasov@edu.hse.ru](mailto:aaatlasov@edu.hse.ru))
- Alina Prolygina ([amprolygina@edu.hse.ru](mailto:amprolygina@edu.hse.ru))

## Quick Start

### [Option 1] Locally With `setup.py`

### Installation Guide

### Prerequisites
- Python 3.11+
- `Docker`

#### Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
pip install .
docker-compose up minio
```

#### Windows
```bash
python -m venv venv
venv/Scripts/activate
pip install .
docker-compose up minio
```

#### Step 1: Run the Dashboard
The dashboard provides a user interface for selecting datasets, configuring model parameters, and visualizing results.
```bash
run-dashboard [--api-type {fastapi,grpc}] [--host HOST] [--port PORT]
```
For fastapi:
```bash
run-dashboard --api-type fastapi --host 127.0.0.1 --port 8000
```
For grpc:
```bash
run-dashboard --api-type grpc --host 127.0.0.1 --port 9090
```
Default is grpc.

#### Step 2: Run the API
The API handles the backend logic for dataset management, model training, and predictions.
```bash
run-server [--api-type {fastapi,grpc}] [--host HOST] [--port PORT] --local
```
For fastapi:
```bash
run-server --api-type fastapi --host 127.0.0.1 --port 8000 --local
```
For grpc:
```bash
run-server --api-type grpc --host 127.0.0.1 --port 9090 --local
```
Default is grpc.

### [Option 2] With `docker-compose`
1. Edit the `docker-compose.yml` `environment` and `ports` fields according to your needs or leave them as is.
2. Build and run:
```bash
docker-compose up --build
```

### Note
- **Dashboard and API run separately**: The dashboard interacts with the backend API through RESTful endpoints.
- **Logging**: Logs for both the API (backend) and the dashboard (client) are written separately for better traceability.
- **Logs** are overwritten on launch of app/dashboard (`w` file modes).
- **Datasets** come preinstalled, but you can add them **if needed**.
- When the dashboard is launched, the **status** of the backend app will appear on the dashboard interface.

## ClearML

For ClearML usage it is neccessary to specify access and secret keys in `docker-compose.yml` (without brackets) and set  `USE_CLEARML=--clearml` (`--no-clearml` for not using).

## Tests

In order to run tests for minio and dataset managing one need to do the following:

```bash
docker-compose up minio
pytest tests/tests.py
```

or just the first command if server is running.

## Features

### Datasets
- The application comes with multiple datasets to choose from (e.g., `moons`, `gauss_quantile`, `xor`, `blobs`).
- Each dataset is displayed immediately after selection.
- User can add or delete their own datasets.

### Models
- A variety of machine learning models are available to choose from, each displayed on the dashboard.
- Help documentation for each model is available on the left sidebar (can be hidden).
- For each model, parameters can be configured in a JSON-like format. Example configuration for Random Forest:
```json
{"n_estimators": 10, "max_depth": 5}
```

Ensure that parameters are enclosed in `{}` brackets.

### Training and Visualization
- Once a model is trained, a probability heatmap is displayed if the model has a `predict_proba` method.
- For SVC model, set `"probability": True` in the configuration to enable probability predictions.

### Prediction
- To predict labels for points, input their features as a list of lists of coma-separated floats (e.g. `[[0.5, 1.5], [-2.3, 2.0]]`).
- You will see predicted labels for corresponding points as text.
- The points will also be displayed on the dashboard amoing training points.

### Additional Actions
- **Model Deletion**: Select a model to delete from the dashboard.
- **Model Retraining**: Select a model to retrain using the same interface as for training.

### Shutdown
To shutdown either dashboard or app just press `Ctrl+C`(Windows, Linux) (`CMD+C`, Mac) in the terminal where you started the app.

### Contact

If you have any questions or issues, please feel free to reach out to the authors:
- Vladislav Bizin ([vkbizin@edu.hse.ru](mailto:vkbizin@edu.hse.ru))
- Alexandr Atlasov ([aaatlasov@edu.hse.ru](mailto:aaatlasov@edu.hse.ru))
- Alina Prolygina ([amprolygina@edu.hse.ru](mailto:amprolygina@edu.hse.ru))
