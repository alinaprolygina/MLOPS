import warnings

warnings.filterwarnings("ignore")
from typing import Dict, List

import psutil
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from mops.backend.apps import parse_arguments
from mops.backend.managers import DatasetManager, ModelManager

logger = logger.bind(logger_name="backend")


def lifespan(MopsApp: FastAPI):
    logger.success("Application has started")
    yield


MopsApp = FastAPI(lifespan=lifespan)


class AvailableDatasetsResponse(BaseModel):
    available_datasets: List[str]


@MopsApp.get("/datasets/available", response_model=AvailableDatasetsResponse)
def list_availabel_datasets():
    """Return the list of available datasets"""
    try:
        available_datasets = DatasetManager.list_available()
        logger.info("Listing available datasets")
        return {"available_datasets": available_datasets}
    except Exception as e:
        logger.error(f"Error listing available datasets: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class UserDatasetsResponse(BaseModel):
    user_datasets: List[str]


@MopsApp.get("/datasets/user_defined", response_model=UserDatasetsResponse)
def list_user_datasets():
    """Return user defined datasets"""
    try:
        user_datasets = DatasetManager.list_user_defined()
        logger.info("Returning the dataset")
        return {"user_datasets": user_datasets}
    except Exception as e:
        logger.error(f"Error returning user defined datasets: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class CreateDatasetRequest(BaseModel):
    dataset_name: str
    X: List[List[float]]
    y: List[int]


class CreateDatasetResponse(BaseModel):
    dataset_name: str


@MopsApp.post("/datasets/create", response_model=CreateDatasetResponse)
def create_dataset(request: CreateDatasetRequest):
    """Add new dataset to database"""
    try:
        logger.info("Creating new dataset")
        dataset_name = DatasetManager.create_dataset(request.dataset_name, request.X, request.y)
        return {"dataset_name": dataset_name}
    except Exception as e:
        logger.error(f"Error creating new dataset: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class DeleteDatasetResponse(BaseModel):
    dataset_name: str


@MopsApp.delete("/datasets/{dataset_name}/delete", response_model=DeleteDatasetResponse)
def delete_dataset(dataset_name: str):
    """Delete user dataset from database"""
    try:
        logger.info(f"Deleting dataset {dataset_name}")
        dataset_name = DatasetManager.delete_dataset(dataset_name)
        return {"dataset_name": dataset_name}
    except Exception as e:
        logger.error(f"Error deleting dataset: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class DatasetResponse(BaseModel):
    X: List[List[float]]
    y: List[float]


@MopsApp.get("/datasets/{dataset_name}", response_model=DatasetResponse)
def get_datasets(dataset_name: str):
    """Return the dataset"""
    try:
        X, y = DatasetManager.load_dataset(dataset_name, as_array=False)
        logger.info("Returning the dataset")
        return {"X": X, "y": y}
    except Exception as e:
        logger.error(f"Error returning the datasets: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class AvailableModelsResponse(BaseModel):
    available_models: List[str]


@MopsApp.get("/models/available", response_model=AvailableModelsResponse)
def list_availabel_models():
    """Return the list of available models"""
    available_models = ModelManager.list_available_models()
    logger.info("Listing available models")
    return {"available_models": available_models}


class TrainedModelsResponse(BaseModel):
    trained_models: List[str]


@MopsApp.get("/models/trained", response_model=TrainedModelsResponse)
def list_trained_models():
    """Return the list of trained models"""
    trained_models = ModelManager.list_trained_models()
    logger.info("Listing trained models")
    return {"trained_models": trained_models}


class ModelDocResponse(BaseModel):
    docstring: str


@MopsApp.get("/models/{model_name}/doc", response_model=ModelDocResponse)
def get_model_docstring(model_name: str):
    """Return the docstring of the model"""
    try:
        docstring = ModelManager.show_doc(model_name)
        logger.info(f"Returning the docstring for '{model_name}'")
        return {"docstring": docstring}
    except Exception as e:
        logger.error(f"Error returning the docstring for '{model_name}': {e}")
        raise HTTPException(status_code=400, detail=str(e))


class TrainedModelDatasetResponse(BaseModel):
    dataset_name: str


@MopsApp.get("/models/{model_id}/dataset_name", response_model=TrainedModelDatasetResponse)
def get_model_dataset_name(model_id: str):
    """Return the name of the dataset that model was trained on"""
    try:
        dataset_name = ModelManager.get_model_dataset_name(model_id)
        logger.info(f"Returning the train dataset name of model '{model_id}'")
        return {"dataset_name": dataset_name}
    except Exception as e:
        logger.error(f"Error returning the train dataset name of model '{model_id}': {e}")
        raise HTTPException(status_code=400, detail=str(e))


class TrainRequest(BaseModel):
    model_name: str
    dataset_name: str
    parameters: Dict = {}


class TrainResponse(BaseModel):
    model_id: str


@MopsApp.post("/models/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    """
    Train a model with specified hyperparameters and return the model ID.
    """
    try:
        X_train, y_train = DatasetManager.load_dataset(request.dataset_name)
        model_id = ModelManager.train_model(
            request.model_name, request.dataset_name, X_train, y_train, **request.parameters
        )
        logger.info(f"Model '{request.model_name}' trained with ID: {model_id}")
        return {"model_id": model_id}
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class PredictRequest(BaseModel):
    model_id: str
    X: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]


@MopsApp.get("/models/{model_id}/predict", response_model=PredictResponse)
def predict(model_id: str, request: PredictRequest):
    """
    Generate predictions for a given model and input data.
    """
    try:
        X = np.array(request.X)
        predictions = ModelManager.predict(model_id, X)
        logger.info(f"Generated predictions for model '{model_id}'")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class PredictProbaRequest(BaseModel):
    model_id: str
    X: List[List[float]]


class PredictProbaResponse(BaseModel):
    predictions: List[float]


@MopsApp.get("/models/{model_id}/predict_proba")
def predict_proba(model_id: str, request: PredictProbaRequest):
    """
    Generate prediction probabilities for a given model and input data.
    """
    try:
        X = np.array(request.X, dtype=float)
        predictions = ModelManager.predict_proba(model_id, X)
        logger.info(f"Generated prediction probabilities for model '{model_id}'")
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class DeleteModelResponse(BaseModel):
    model_id: str


@MopsApp.delete("/models/{model_id}/delete", response_model=DeleteModelResponse)
def delete_model(model_id: str):
    """
    Delete a trained model by its model ID.
    """
    try:
        ModelManager.delete_model(model_id)
        logger.info(f"Model '{model_id}' deleted")
        return {"model_id": model_id}
    except ValueError as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class RetrainRequest(BaseModel):
    dataset_name: str
    parameters: Dict = {}


class RetrainResponse(BaseModel):
    model_id: str


@MopsApp.post("/models/{model_id}/retrain", response_model=RetrainResponse)
def retrain_model(model_id: str, request: RetrainRequest):
    """
    Re-train an already trained model.
    """
    try:
        X_train, y_train = DatasetManager.load_dataset(request.dataset_name)
        ModelManager.retrain_model(model_id, request.dataset_name, X_train, y_train)
        logger.info(f"Model '{model_id}' re-trained successfully")
        return {"model_id": model_id}
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class MemoryUsage(BaseModel):
    total: int
    used: int
    free: int
    percent: float


class HealthResponse(BaseModel):
    cpu_usage_percent: float
    memory_usage: MemoryUsage


@MopsApp.get("/health", response_model=HealthResponse)
def check_health():
    """
    Health check endpoint to verify the service status.
    """
    logger.info("Health check called")
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = {
        "total": memory_info.total,
        "used": memory_info.used,
        "free": memory_info.free,
        "percent": memory_info.percent,
    }
    return {
        "cpu_usage_percent": cpu_usage,
        "memory_usage": memory_usage,
    }


def serve(host: str, port: int):
    uvicorn.run("mops.backend.apps.fastapi.app:MopsApp", host=host, port=port)


def main():
    args = parse_arguments("FastAPI")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
