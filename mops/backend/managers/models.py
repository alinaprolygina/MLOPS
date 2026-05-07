from io import BytesIO
from random import choices
from collections import defaultdict
from string import ascii_lowercase, digits
from typing import Any, List, Iterable, Dict, Tuple

import os 
import joblib
import numpy as np
import pandas as pd
from clearml import Task
from loguru import logger
from mops import parse_arguments

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logger.bind(name="backend")


def genid(length: int = 6) -> str:
    return "".join(choices(ascii_lowercase + digits, k=length))


def gen_unq_modelid(model_name: str, dataset_name: str, model_ids: Iterable[str], length: int = 6) -> Tuple[str, str]:
    while True:
        random_id = genid(length)
        model_id = f"{model_name}_{dataset_name}_{random_id}"
        if model_id not in model_ids:
            return random_id, model_id


class ModelManager:
    """
    Class that handles model logic, such as training, loading, and deleting models.
    """

    clearml: bool = False
    s3_client = None
    bucket_name: str = ""
    models: Dict[str, any] = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
        "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
        "LogisticRegression": LogisticRegression,
        "GaussianNB": GaussianNB,
        "BernoulliNB": BernoulliNB,
        "KNeighborsClassifier": KNeighborsClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "CatBoostClassifier": CatBoostClassifier,
        "XGBClassifier": XGBClassifier,
        "SVC": SVC,
    }
    trained_models: defaultdict = defaultdict(dict)

    @staticmethod
    def ensure_bucket_exists():
        try:
            ModelManager.s3_client.head_bucket(Bucket=ModelManager.bucket_name)
            logger.info(f"Bucket '{ModelManager.bucket_name}' already exists.")
        except ModelManager.s3_client.exceptions.ClientError:
            logger.warning(f"Bucket '{ModelManager.bucket_name}' does not exist. Creating...")
            ModelManager.s3_client.create_bucket(Bucket=ModelManager.bucket_name)
            logger.success(f"Bucket '{ModelManager.bucket_name}' created.")

    @staticmethod
    def parse_bucket():
        """
        Parses existing models in MinIO bucket.
        """
        logger.info("Parsing trained models in MinIO bucket")
        try:
            response = ModelManager.s3_client.list_objects_v2(Bucket=ModelManager.bucket_name)
            if "Contents" in response:
                for obj in response["Contents"]:
                    model_id = obj["Key"].split(".")[0]
                    ModelManager.trained_models[model_id]["path"] = obj["Key"]
                    logger.info(f"Found trained model: {model_id}")
        except ModelManager.s3_client.exceptions.NoSuchBucket:
            logger.warning(f"Bucket '{ModelManager.bucket_name}' does not exist. Creating...")
            ModelManager.s3_client.create_bucket(Bucket=ModelManager.bucket_name)
            logger.success(f"Bucket '{ModelManager.bucket_name}' created.")

    @staticmethod
    def setup(
        s3_client,
        clearml: bool,
        bucket_name: str,
    ):
        """
        Configures the ModelManager with the required S3 settings.

        This method initializes the ModelManager's static properties, ensuring that
        the specified S3 bucket exists and parsing its contents for further operations.

        Args:
            s3_client (boto3.s3_client): An initialized S3 client object used for interacting with S3.
            bucket_name (str): The name of the S3 bucket to be used for storing models.

        Actions:
            - Sets the static attributes of the ModelManager class.
            - Ensures that the specified S3 bucket exists.
            - Parses the contents of the S3 bucket.
        """
        ModelManager.s3_client = s3_client
        ModelManager.bucket_name = bucket_name
        ModelManager.clearml = clearml

        if ModelManager.clearml:
            Task.set_credentials(
            api_host=os.getenv("CLEARML_API_HOST"),
            files_host=os.getenv("CLEARML_FILES_HOST"),
            web_host=os.getenv("CLEARML_WEB_HOST"),
            key=os.getenv("CLEARML_API_ACCESS_KEY"),
            secret=os.getenv("CLEARML_API_SECRET_KEY"),
        )

        ModelManager.ensure_bucket_exists()
        ModelManager.parse_bucket()

    @staticmethod
    def show_doc(model_name: str) -> str:
        """
        Returns the docstring of a specified model class.

        Parameters
        ----------
        model_name : str
            The name of the model to get the docstring for.

        Returns
        -------
        str
            The docstring of the specified model class.

        Raises
        ------
        ValueError
            If the model is not available.
        """
        logger.info(f"Retrieving docstring for model '{model_name}'")
        if model_name not in ModelManager.models:
            err_msg = f"Model '{model_name}' is not available. Available 'model_name' values: {', '.join(ModelManager.models.keys())}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        logger.success(f"Docstring for model '{model_name}' retrieved successfully")
        return ModelManager.models[model_name].__doc__

    @staticmethod
    def train_model(
        model_name: str,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **parameters: Any,
    ) -> str:
        """
        Trains a model with the given parameters and saves it.

        Parameters
        ----------
        model_name : str
            The name of the model to train.
        dataset_name : str
            The name of the dataset used for training.
        X_train : np.ndarray
            Training features (2D array).
        y_train : np.ndarray
            Training labels (1D array).
        parameters : dict
            Additional parameters to pass to the model.

        Returns
        -------
        str
            The unique ID of the trained model.

        Raises
        ------
        ValueError
            If the model is not available.
        """
        random_id, model_id = gen_unq_modelid(model_name, dataset_name, ModelManager.trained_models.keys())

        if ModelManager.clearml:
            task = Task.init(
                project_name="MLOps Project",
                task_name=f"Train {model_name} on {dataset_name}, (id={random_id})",
                task_type=Task.TaskTypes.training,
            )
            task.connect({"model_id": model_id})
            task.connect(parameters)

            task.connect({"dataset_name": dataset_name})
            task.upload_artifact(name='train.data', artifact_object=X_train)
            task.upload_artifact(name='train.target', artifact_object=y_train)
            task.upload_artifact(
                name='train.data.eda', 
                artifact_object=pd.DataFrame(X_train).describe(include=np.number),
            )
            task.upload_artifact(
                name='train.target.eda', 
                artifact_object=pd.DataFrame(y_train.reshape(-1, 1)).describe(include=np.number),
            )

        logger.info(f"Training model {model_name} on {dataset_name} with parameters: {parameters}")
        if model_name not in ModelManager.models:
            err_msg = f"Model '{model_name}' is not available. Available 'model_name' values: {', '.join(ModelManager.models.keys())}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        model_class = ModelManager.models[model_name]
        model = model_class(**parameters)

        model.fit(X_train, y_train)
        logger.success("Model trained successfully")
        if ModelManager.clearml:
            task.get_logger().report_scalar("Training", "completion", iteration=1, value=100)

        ModelManager.trained_models[model_id]["dataset_name"] = dataset_name

        ModelManager.save_model(model_id, model)
        if ModelManager.clearml:
            task.upload_artifact("trained_model", model)

            task.close()
        return model_id

    @staticmethod
    def save_model(model_id: str, model) -> None:
        """
        Saves the model to MinIO.

        Parameters
        ----------
        model_id: str
            ID of the model
        model:
            Trained model object
        """
        logger.info("Saving model to MinIO")

        # Serialize the model into memory
        model_data = BytesIO()
        joblib.dump(model, model_data)
        model_data.seek(0)

        # Upload to MinIO
        s3_key = f"{model_id}.joblib"
        ModelManager.s3_client.put_object(Bucket=ModelManager.bucket_name, Key=s3_key, Body=model_data.getvalue())
        ModelManager.trained_models[model_id]["path"] = s3_key
        logger.success(f"Model saved to MinIO with id: {model_id}")

    @staticmethod
    def list_trained_models() -> List[str]:
        """
        Lists the IDs of all trained models.

        Returns
        -------
        List[str]
            A list of trained model IDs.
        """
        logger.info("Listing available models")
        return list(ModelManager.trained_models.keys())

    @staticmethod
    def list_available_models() -> List[str]:
        """
        Lists the names of available model classes for training.

        Returns
        -------
        List[str]
            A list of available model names.
        """
        logger.info("Listing trained models")
        return list(ModelManager.models.keys())

    @staticmethod
    def load_model(model_id: str) -> BaseEstimator:
        """
        Loads a previously trained model by its ID from MinIO.

        Parameters
        ----------
        model_id : str
            The ID of the model to load.

        Returns
        -------
        BaseEstimator
            The loaded model object.
        """
        logger.info(f"Loading model with id: {model_id} from MinIO")
        if model_id not in ModelManager.trained_models:
            err_msg = f"Model '{model_id}' does not exist"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Download model from MinIO
        s3_key = ModelManager.trained_models[model_id]["path"]
        response = ModelManager.s3_client.get_object(Bucket=ModelManager.bucket_name, Key=s3_key)
        model_data = BytesIO(response["Body"].read())

        # Deserialize the model
        model = joblib.load(model_data)
        logger.success(f"Loaded model with id: {model_id}")
        return model

    @staticmethod
    def get_model_dataset_name(model_id: str) -> str:
        """
        Retrieves the name of the dataset used to train a specific model.

        Parameters
        ----------
        model_id : str
            The ID of the model.

        Returns
        -------
        str
            The name of the dataset used for training the model.

        Raises
        ------
        ValueError
            If the model ID does not exist.
        """
        logger.info(f"Retrieving training dataset nam for model: {model_id}")
        if model_id not in ModelManager.trained_models:
            err_msg = f"Model '{model_id}' does not exist"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return ModelManager.trained_models[model_id]["dataset_name"]

    @staticmethod
    def validate_input(X: List[List[float]]) -> None:
        """
        Validates the input

        Parameters
        ----------
        dataset_name (str):
            name of new dataset
        X (List[List[float]]):
            data to validate

        Raises
        ------
        ValueError
            If X can not be converted to np.array or X and y have different lengths.
        """
        logger.info("Validating data types")
        try:
            np.array(X)
        except ValueError as e:
            err_msg = "X failed type validation: can not be converted to np.array"
            logger.error(err_msg)
            raise ValueError(err_msg) from e

        logger.success("Validation successfull")

    @staticmethod
    def predict(model_id: str, X: List[List[float]]) -> List[int]:
        """
        Generates predictions using a trained model.

        Parameters
        ----------
        model_id : str
            The ID of the trained model.
        X : np.ndarray
            Features for which to generate predictions.

        Returns
        -------
        List[int]
            A list of predictions.

        Raises
        ------
        ValueError
            If the model ID does not exist.
        """
        ModelManager.validate_input(X)
        logger.info(f"Making predicions with model: {model_id}")
        model = ModelManager.load_model(model_id)
        predictions = model.predict(np.array(X, dtype=float))
        logger.success("Successfully got predictions from model")
        return predictions.tolist()

    @staticmethod
    def predict_proba(model_id: str, X: List[List[float]]) -> List[float]:
        """
        Generates prediction probabilities using a trained model.

        Parameters
        ----------
        model_id : str
            The ID of the trained model.
        X : np.ndarray
            Features for which to generate prediction probabilities.

        Returns
        -------
        List[float]
            A list of prediction probabilities.

        Raises
        ------
        ValueError
            If the model ID does not exist.
        """
        ModelManager.validate_input(X)
        model = ModelManager.load_model(model_id)
        probabilities = model.predict_proba(np.array(X, dtype=float))[:, 1]
        return probabilities.tolist()

    @staticmethod
    def delete_model(model_id: str) -> None:
        """
        Deletes a trained model from MinIO.

        Parameters
        ----------
        model_id : str
            The ID of the model to delete.
        """
        logger.info(f"Deleting model: {model_id} from MinIO")
        if model_id not in ModelManager.trained_models:
            err_msg = f"Model '{model_id}' does not exist"
            logger.error(err_msg)
            raise ValueError(err_msg)

        s3_key = ModelManager.trained_models.pop(model_id)["path"]
        ModelManager.s3_client.delete_object(Bucket=ModelManager.bucket_name, Key=s3_key)
        logger.success(f"Successfully deleted model with id: {model_id}")

    @staticmethod
    def retrain_model(
        model_id: str,
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **parameters: Any,
    ) -> None:
        """
        Re-trains an already existing model with new data.

        Parameters
        ----------
        model_id : str
            The ID of the model to retrain.
        dataset_name : str
            The name of the dataset used for re-training.
        X_train : np.ndarray
            Training features (2D array or DataFrame).
        y_train : np.ndarray
            Training labels (1D array or DataFrame).
        parameters : dict
            Additional parameters to pass to the model.

        Raises
        ------
        ValueError
            If the model ID does not exist.
        """
        if ModelManager.clearml:
            task = Task.init(
                project_name="ML Training Project",
                task_name=f"Retrain {model_id} on {dataset_name} (id={model_id.split('_')[-1]})",
                task_type=Task.TaskTypes.training
            )
            task.connect({"model_id": model_id})
            task.connect(parameters)

            task.connect({"dataset_name": dataset_name})
            task.upload_artifact(name='train.data', artifact_object=X_train)
            task.upload_artifact(name='train.target', artifact_object=y_train)
            task.upload_artifact(
                name='train.data.eda', 
                artifact_object=pd.DataFrame(X_train).describe(include=np.number),
            )
            task.upload_artifact(
                name='train.target.eda', 
                artifact_object=pd.DataFrame(y_train.reshape(-1, 1)).describe(include=np.number),
            )

        logger.info(f"Training model {model_id} on {dataset_name} with parameters: {parameters}")
        if model_id not in ModelManager.trained_models:
            err_msg = f"Model '{model_id}' does not exist"
            logger.error(err_msg)
            raise ValueError(err_msg)

        model = ModelManager.load_model(model_id)
        model.set_params(**parameters)
        model.fit(X_train, y_train)

        ModelManager.save_model(model_id, model)
        if ModelManager.clearml:
            task.upload_artifact("trained_model", model)
        ModelManager.trained_models[model_id]["dataset_name"] = dataset_name
        if ModelManager.clearml:
            task.close()
