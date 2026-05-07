import ast
from concurrent import futures

import grpc
import psutil
import numpy as np
from loguru import logger

from mops.backend.apps import parse_arguments
import mops.backend.apps.grpc.server_pb2 as server_proto
import mops.backend.apps.grpc.server_pb2_grpc as server_grpc
from mops.backend.managers import DatasetManager, ModelManager

logger = logger.bind(logger_name="backend")


class MopsApp(server_grpc.MopsAppServicer):
    """
    Service class
    """

    def __init__(self):
        pass

    @staticmethod
    def HealthCheck(request, context):
        """
        Health check endpoint to verify the service status.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = {
            "total": memory_info.total,
            "used": memory_info.used,
            "free": memory_info.free,
            "percent": memory_info.percent,
        }
        return server_proto.ServerStatus(cpu=cpu_usage, mem=memory_usage)

    @staticmethod
    def List_trained_models(request, context):
        """
        Return the list of trained models.
        """
        try:
            model_list = ModelManager.list_trained_models()
            return server_proto.ModelList(models=model_list)
        except Exception as e:
            logger.error(f"Failed to list available datasets : {e}")
            context.set_details(str(e))

    @staticmethod
    def List_available_models(request, context):
        """
        Return a list of model classes available for training.
        """
        try:
            model_list = ModelManager.list_available_models()
            return server_proto.AvailableModels(models=model_list)
        except Exception as e:
            logger.error(f"Failed to list available datasets: {e}")
            context.set_details(str(e))

    @staticmethod
    def List_available_datasets(request, context):
        """
        Return the list of available datasets.
        """
        try:
            available_datasets = DatasetManager.list_available()
            return server_proto.AvailableDatasets(datasets=available_datasets)
        except Exception as e:
            logger.error(f"Failed to list available datasets: {e}")
            context.set_details(str(e))

    @staticmethod
    def List_user_datasets(request, context):
        """
        Return the list of available datasets.
        """
        try:
            user_datasets = DatasetManager.list_user_defined()
            return server_proto.UserDatasetsOutput(datasets=user_datasets)
        except Exception as e:
            logger.error(f"Failed to list user defined datasets: {e}")
            context.set_details(str(e))

    @staticmethod
    def CreateDataset(request, context):
        """
        Saves a dataset that user creates by himself
        """
        try:
            logger.info(f"Server creating dataset {request.dataset_name}.")
            X = [[value for value in row.values] for row in request.X.rows]
            y = [value for value in request.y.values]
            data_name = DatasetManager.create_dataset(request.dataset_name, X, y)
            logger.success(f"Created dataset {data_name}.")
            return server_proto.CreateDatasetOutput(dataset_name=data_name)
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}.")
            context.set_details(str(e))

    @staticmethod
    def DeleteDataset(request, context):
        """
        Deletes user's dataset by its name
        """
        try:
            logger.info(f"Deleting dataset {request.dataset_name}.")
            deleted_name = DatasetManager.delete_dataset(request.dataset_name)
            logger.success(f"Dataset {request.dataset_name} has been deleted.")
            return server_proto.DeleteDatasetOutput(dataset_name=deleted_name)
        except Exception as e:
            logger.error(f"Failed to delete dataset {request.dataset_name}: {e}.")
            context.set_details(str(e))

    @staticmethod
    def GetDataset(request, context):
        """
        Get dataset by its name.
        """
        try:
            X, y = DatasetManager.load_dataset(request.dataset_name)
            array_2d = server_proto.Array2Df(rows=[server_proto.Array1Df(values=row.tolist()) for row in X])
            array_1d = server_proto.Array1Di(values=y.tolist())
            return server_proto.Dataset(X=array_2d, y=array_1d)
        except Exception as e:
            logger.error(f"Failed to load dataset {request.dataset_name}: {e}")
            context.set_details(str(e))

    @staticmethod
    def GetModelsDataset(request, context):
        """
        Returns datasetname that was used for a model's training.
        """
        try:
            dataname = ModelManager.get_model_dataset_name(request.model_id)
            return server_proto.DatasetName(dataset=dataname)
        except Exception as e:
            logger.error(f"Failed to show dataset for model {request.model_name}: {e}")
            context.set_details(str(e))

    @staticmethod
    def Train(request, context):
        """
        Train a model with specified hyperparameters and return the model ID.
        """
        try:
            X, y = DatasetManager.load_dataset(request.dataset_name)
            parameters = ast.literal_eval(request.parameters)
            model_id = ModelManager.train_model(request.model_name, request.dataset_name, X, y, **parameters)
            return server_proto.TrainOutput(
                message=f"Model {request.model_name} with id {model_id} has been trained and saved", model_id=model_id
            )
        except Exception as e:
            logger.error(f"Failed to train model {request.model_name} on dataset {request.dataset_name} : {e}")
            context.set_details(str(e))

    @staticmethod
    def Predict(request, context):
        """
        Generate predictions for a given model and input data.
        """
        try:
            X = np.array([[value for value in row.values] for row in request.X.rows], dtype=float)
            preds = ModelManager.predict(request.model_id, X)
            return server_proto.PredictOutput(prediction=preds)
        except Exception as e:
            logger.error(f"Model {request.model_id} failed to predict: {e}")
            context.set_details(str(e))

    @staticmethod
    def PredictProba(request, context):
        """
        Generate probability predictions for a given model and input data.
        """
        try:
            X = np.array([[value for value in row.values] for row in request.X.rows], dtype=float)
            predictions = ModelManager.predict_proba(request.model_id, X)
            return server_proto.PredictProbaOutput(probas=predictions)
        except Exception as e:
            logger.error(f"Error during probas prediction: {e}")
            context.set_details(str(e))

    @staticmethod
    def Retrain(request, context):
        """
        Re-train an already existing model with new data.
        """
        try:
            X, y = DatasetManager.load_dataset(request.dataset_name)
            parameters = ast.literal_eval(request.parameters)
            ModelManager.retrain_model(request.model_id, request.dataset_name, X, y, **parameters)
            return server_proto.RetrainOutput(model_id=request.model_id)
        except Exception as e:
            logger.error(f"Failed to re-train model {request.model_id}: {e}")
            context.set_details(str(e))

    @staticmethod
    def DeleteModel(request, context):
        """
        Delete a trained model.
        """
        try:
            ModelManager.delete_model(request.model_id)
            logger.success(f"Model {request.model_id} has been deleted")
            return server_proto.DeleteModelOutput(model_id=request.model_id)
        except Exception as e:
            logger.error(f"Failed to delete model {request.model_id}: {e}")
            context.set_details(str(e))

    @staticmethod
    def ShowDocString(request, context):
        """
        Returns models docstring by its id.
        """
        try:
            docstr = ModelManager.show_doc(request.model_id)
            return server_proto.ModelDocString(docsting=docstr)
        except Exception as e:
            logger.error(f"Failed to show doctring for model {request.model_id}: {e}")
            context.set_details(str(e))


def serve(host: str, port: int):
    logger.info(f"Starging gRPC server at {host}:{port}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server_grpc.add_MopsAppServicer_to_server(MopsApp, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.success("Server started")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(None)
        logger.success("Server stopped")


def main():
    args = parse_arguments("gRPC")
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
