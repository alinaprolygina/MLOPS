from typing import Literal, Dict, List, Any, Tuple

import numpy as np
import requests
import grpc
from loguru import logger

import mops.backend.apps.grpc.server_pb2 as server_proto
from mops.backend.apps.grpc.server_pb2_grpc import MopsAppStub

logger = logger.bind(logger_name="frontend")


class Client:
    api_type: Literal["grpc", "fastapi"] = None
    host: str = None
    port: int = None

    base_url: str = None
    session: requests.Session = None

    grpc_channel: grpc.Channel = None
    grpc_stub: MopsAppStub = None

    def __init__(self):
        pass

    @staticmethod
    def connect():
        if Client.api_type == "fastapi":
            Client.session = requests.Session()
            Client.base_url = f"http://{Client.host}:{Client.port}"

        elif Client.api_type == "grpc":
            try:
                Client.grpc_channel = grpc.insecure_channel(f"{Client.host}:{Client.port}")
                Client.grpc_stub = MopsAppStub(Client.grpc_channel)
            except grpc.RpcError as e:
                raise ConnectionError(f"Failed to connect to gRPC server at {Client.host}:{Client.port}: {e}")

    @staticmethod
    def _handle_fastapi_request(method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        try:
            url = f"{Client.base_url}/{endpoint.lstrip('/')}"
            response = getattr(Client.session, method.lower())(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            err_msg = f"FastAPI request failed: {e}"
            logger.error(err_msg)
            raise
        except Exception as e:
            details = response.json().get("detail", "")
            err_msg = f"FastAPI request failed: {e}. Details: {details}"
            logger.error(err_msg)
            raise type(e)(err_msg) from e

    @staticmethod
    def get_health() -> Dict[str, Any]:
        if Client.api_type == "fastapi":
            return Client._handle_fastapi_request("get", "health")
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.HealthCheck(server_proto.Empty4())
                return {
                    "cpu_usage_percent": response.cpu,
                    "memory_usage": response.mem,
                }
            except Exception as e:
                err_msg = f"gRPC `HealthCheck` failed: {e}"
                logger.error(err_msg)
                raise ConnectionError(err_msg)

    @staticmethod
    def create_dataset(dataset_name: str, X: List[List[float]], y: List[int]) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request(
                "post",
                "datasets/create",
                json={"dataset_name": dataset_name, "X": X, "y": y},
            )
            return response.get("dataset_name", "")
        elif Client.api_type == "grpc":
            try:
                array_2d = server_proto.Array2Df(rows=[server_proto.Array1Df(values=row) for row in X])
                array_1d = server_proto.Array1Di(values=y)
                response = Client.grpc_stub.CreateDataset(
                    server_proto.UserDataset(X=array_2d, y=array_1d, dataset_name=dataset_name)
                )
                return response.dataset_name
            except Exception as e:
                err_msg = f"gRPC `CreateDataset` failed: {e}"
                logger.error(err_msg)
                raise ConnectionError(err_msg)

    @staticmethod
    def delete_dataset(dataset_name: str):
        if Client.api_type == "fastapi":
            return Client._handle_fastapi_request("delete", f"datasets/{dataset_name}/delete").get("dataset_name", "")
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.DeleteDataset(server_proto.DeleteDatasetInput(dataset_name=dataset_name))
                return response.dataset_name
            except Exception as e:
                err_msg = f"gRPC `DeleteDataset` failed: {e}"
                logger.error(err_msg)
                raise ConnectionError(err_msg)

    @staticmethod
    def get_available_datasets() -> List[str]:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", "datasets/available")
            return response.get("available_datasets", [])
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.List_available_datasets(server_proto.Empty1())
                return list(response.datasets)
            except Exception as e:
                raise ConnectionError(f"gRPC `ListAvailableDatasets` failed: {e}")

    @staticmethod
    def get_user_datasets() -> List[str]:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", "datasets/user_defined")
            return response.get("user_datasets", [])
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.List_user_datasets(server_proto.UserDatasetsInput())
                return list(response.datasets)
            except Exception as e:
                raise ConnectionError(f"gRPC `ListUserDatasets` failed: {e}")

    @staticmethod
    def get_available_models() -> List[str]:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", "models/available")
            return response.get("available_models", [])
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.List_available_models(server_proto.Empty1())
                return list(response.models)
            except Exception as e:
                raise ConnectionError(f"gRPC `ListAvailableModels` failed: {e}")

    @staticmethod
    def get_docstring(model_name: str) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", f"models/{model_name}/doc")
            return response.get("docstring", "")
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.ShowDocString(server_proto.Model(model_id=model_name))
                return response.docsting
            except Exception as e:
                raise ConnectionError(f"gRPC `ShowDocString` failed: {e}")

    @staticmethod
    def get_trained_models() -> List[str]:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", "models/trained")
            return response.get("trained_models", [])
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.List_trained_models(server_proto.Empty1())
                return list(response.models)
            except Exception as e:
                raise ConnectionError(f"gRPC `ListAvailableModels` failed: {e}")

    @staticmethod
    def get_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", f"datasets/{dataset_name}")
            X = np.array(response.get("X", [[]]), dtype=float)
            y = np.array(response.get("y", []), dtype=int)
            return X, y
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.GetDataset(server_proto.GetDataName(dataset_name=dataset_name))
                X = np.array([[value for value in row.values] for row in response.X.rows], dtype=float)
                y = np.array([value for value in response.y.values], dtype=int)
                return X, y
            except Exception as e:
                raise ConnectionError(f"gRPC `GetDataset` failed: {e}")

    @staticmethod
    def get_model_dataset(model_id: str) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("get", f"models/{model_id}/dataset_name")
            return response.get("dataset_name", "")
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.GetModelsDataset(server_proto.ModelID(model_id=model_id))
                return response.dataset
            except Exception as e:
                raise ConnectionError(f"gRPC `GetModelsDataset` failed: {e}")

    @staticmethod
    def train_model(model_name: str, dataset_name: str, parameters: Dict[str, Any] = {}) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request(
                "post",
                "models/train",
                json={"model_name": model_name, "dataset_name": dataset_name, "parameters": parameters},
            )
            return response.get("model_id", "")
        elif Client.api_type == "grpc":
            try:
                request = server_proto.TrainInput(
                    model_name=model_name, dataset_name=dataset_name, parameters=str(parameters)
                )
                response = Client.grpc_stub.Train(request)
                return response.model_id
            except Exception as e:
                raise ConnectionError(f"gRPC `Train` failed: {e}")

    @staticmethod
    def retrain_model(model_id: str, dataset_name: str, parameters: Dict[str, Any] = {}) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request(
                "post",
                f"models/{model_id}/retrain",
                json={"model_id": model_id, "dataset_name": dataset_name, "parameters": parameters},
            )
            return response.get("model_id", "")
        elif Client.api_type == "grpc":
            try:
                request = server_proto.RetrainInput(
                    model_id=model_id, dataset_name=dataset_name, parameters=str(parameters)
                )
                response = Client.grpc_stub.Retrain(request)
                return response.model_id
            except Exception as e:
                raise ConnectionError(f"gRPC `Retrain` failed: {e}")

    @staticmethod
    def get_predict(model_id: str, X: List[List[float]]) -> np.ndarray:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request(
                "get", f"models/{model_id}/predict", json={"model_id": model_id, "X": X}
            )
            y = response.get("predictions", [])
            return np.array(y, dtype=int)
        elif Client.api_type == "grpc":
            try:
                request = server_proto.PredictInput(
                    model_id=model_id, X=server_proto.Array2Df(rows=[server_proto.Array1Df(values=row) for row in X])
                )
                response = Client.grpc_stub.Predict(request)
                y = response.prediction
                return np.array(y, dtype=int)
            except Exception as e:
                raise ConnectionError(f"gRPC `Predict` failed: {e}")

    @staticmethod
    def get_predict_proba(model_id: str, X: List[List[float]]) -> np.ndarray:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request(
                "get", f"models/{model_id}/predict_proba", json={"model_id": model_id, "X": X}
            )
            y = response.get("predictions", [])
            return np.array(y, dtype=float)
        elif Client.api_type == "grpc":
            try:
                request = server_proto.PredictInput(
                    model_id=model_id, X=server_proto.Array2Df(rows=[server_proto.Array1Df(values=row) for row in X])
                )
                response = Client.grpc_stub.PredictProba(request)
                y = response.probas
                return np.array(y, dtype=float)
            except Exception as e:
                raise ConnectionError(f"gRPC `PredictProba` failed: {e}")

    @staticmethod
    def delete_model(model_id: str) -> str:
        if Client.api_type == "fastapi":
            response = Client._handle_fastapi_request("delete", f"/models/{model_id}/delete")
            return response.get("model_id", "")
        elif Client.api_type == "grpc":
            try:
                response = Client.grpc_stub.DeleteModel(server_proto.DeleteModelInput(model_id=model_id))
                return response.model_id
            except Exception as e:
                raise ConnectionError(f"gRPC `Delete` failed: {e}")
