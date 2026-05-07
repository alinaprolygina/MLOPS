import os
from typing import List, Tuple, Type, Set

import numpy as np
from io import BytesIO
from loguru import logger

logger = logger.bind(logger_name="backend")


class DatasetManager:
    """
    Class responsible for handling dataset operations such as loading and converting datasets.
    """

    s3_client = None
    bucket_name: str = ""
    local_datasets_path: str = ""
    names: Set[str] = set()
    base_datasets: Set[str] = set()

    @staticmethod
    def ensure_bucket_exists():
        """
        Ensures that the bucket exists in MinIO. Creates the bucket if it doesn't exist and uploads datasets.
        """
        try:
            DatasetManager.s3_client.head_bucket(Bucket=DatasetManager.bucket_name)
            logger.info(f"Bucket '{DatasetManager.bucket_name}' already exists.")
        except DatasetManager.s3_client.exceptions.ClientError:
            logger.warning(f"Bucket '{DatasetManager.bucket_name}' does not exist. Creating...")
            DatasetManager.s3_client.create_bucket(Bucket=DatasetManager.bucket_name)
            logger.success(f"Bucket '{DatasetManager.bucket_name}' created.")

            logger.info("Uploading datasets to MinIO...")
            DatasetManager.upload_datasets_to_minio(DatasetManager.local_datasets_path)
            logger.success("Datasets uploaded successfully.")

    @staticmethod
    def upload_datasets_to_minio(local_path):
        """
        Uploads all datasets from a local directory to MinIO.

        Parameters
        ----------
        local_path : str
            Path to the local directory containing datasets.
        """
        if not os.path.exists(local_path):
            logger.error(f"Local directory '{local_path}' does not exist. Ensure the path is correct.")
            return

        logger.info(f"Uploading datasets from '{local_path}' to MinIO...")
        for dataset_name in os.listdir(local_path):
            dataset_path = os.path.join(local_path, dataset_name)
            if os.path.isdir(dataset_path):
                logger.info(f"Processing dataset '{dataset_name}'...")
                for file_name in os.listdir(dataset_path):
                    file_path = os.path.join(dataset_path, file_name)
                    s3_key = f"{dataset_name}/{file_name}"

                    try:
                        logger.info(f"Uploading '{file_name}' as '{s3_key}'...")
                        DatasetManager.s3_client.upload_file(file_path, DatasetManager.bucket_name, s3_key)
                        logger.success(f"Uploaded '{file_name}' to bucket '{DatasetManager.bucket_name}' as '{s3_key}'")
                    except Exception as e:
                        logger.error(f"Error uploading '{file_name}': {e}")

    @staticmethod
    def parse_bucket():
        """
        Parses existing datasets in MinIO bucket.
        """
        logger.info("Parsing datasets in MinIO bucket")
        try:
            response = DatasetManager.s3_client.list_objects_v2(Bucket=DatasetManager.bucket_name)
            if "Contents" in response:
                for obj in response["Contents"]:
                    dataset_name = obj["Key"].split("/")[0]  # Имя датасета (первая часть ключа)
                    DatasetManager.names.add(dataset_name)
                    logger.info(f"Found dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error parsing datasets: {e}")

    @staticmethod
    def setup(s3_client, bucket_name: str, local_datasets_path: str, names: Set[str], base_datasets: Set[str]):
        """
        Configures the DatasetManager with the required S3 and local dataset settings.

        This method initializes the DatasetManager's static properties, ensuring that
        the S3 bucket exists and parsing its contents for further operations.

        Parameters:
            s3_client (boto3.s3_client): An initialized S3 client object used for interacting with S3.
            bucket_name (str): The name of the S3 bucket to be used for storing datasets.
            local_datasets_path (str): The path to the local directory where datasets are stored.
            names (Set[str]): A set of dataset names to be managed.
            base_datasets (Set[str]): A set of base datasets to be used as references.

        Actions:
            - Sets the static attributes of the DatasetManager class.
            - Ensures that the specified S3 bucket exists.
            - Parses the contents of the S3 bucket.
        """
        DatasetManager.s3_client = s3_client
        DatasetManager.bucket_name = bucket_name
        DatasetManager.local_datasets_path = local_datasets_path
        DatasetManager.names = names
        DatasetManager.base_datasets = base_datasets

        DatasetManager.ensure_bucket_exists()
        DatasetManager.parse_bucket()

    @staticmethod
    def load_dataset(dataset_name: str, as_array: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads the features (X) and labels (y) for a given dataset from MinIO.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.
        as_array : bool
            Whether to return as np.array. If true: np.array, else lists.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the features (X) and labels (y) of the dataset.
        """
        if dataset_name not in DatasetManager.names:
            raise ValueError(f"Dataset '{dataset_name}' not found in MinIO.")

        logger.info(f"Loading dataset '{dataset_name}' from MinIO")
        try:
            X_response = DatasetManager.s3_client.get_object(
                Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/X_train.npy"
            )
            y_response = DatasetManager.s3_client.get_object(
                Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/y_train.npy"
            )
            X = np.load(BytesIO(X_response["Body"].read()))
            y = np.load(BytesIO(y_response["Body"].read()))
            logger.success(f"Dataset '{dataset_name}' successfully loaded.")
            if as_array:
                return X, y
            return X.tolist(), y.tolist()
        except Exception as e:
            logger.error(f"Error loading dataset '{dataset_name}': {e}")
            raise

    @staticmethod
    def validate_dataset(X: List[List[float]], y: List[int]) -> None:
        """
        Validates the dataset for correct types and lengths.

        Parameters
        ----------
        X : List[List[float]]
            Training data.
        y : List[int]
            Target labels.

        Raises
        ------
        ValueError
            If X or y is not valid.
        """
        logger.info("Validating dataset")
        try:
            np.array(X)
        except ValueError as e:
            raise ValueError("X must be convertible to np.array") from e

        if not isinstance(y, list) or not all(isinstance(val, int) for val in y):
            raise TypeError("y must be a list of integers")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        logger.success("Dataset validation successful")

    @staticmethod
    def create_dataset(dataset_name: str, X: List[List[float]], y: List[int]) -> str:
        """
        Creates and saves a dataset to MinIO.

        Parameters
        ----------
        dataset_name : str
            Name of the new dataset.
        X : List[List[float]]
            Training data.
        y : List[int]
            Target labels.

        Returns
        -------
        str
            Name of the created dataset.
        """
        DatasetManager.ensure_bucket_exists()

        if dataset_name in DatasetManager.names:
            raise ValueError(f"Dataset '{dataset_name}' already exists in MinIO.")

        DatasetManager.validate_dataset(X, y)

        logger.info(f"Saving dataset '{dataset_name}' to MinIO")
        try:
            # Convert X and y to numpy and save as BytesIO
            X_data = BytesIO()
            y_data = BytesIO()
            np.save(X_data, np.array(X))
            np.save(y_data, np.array(y))
            X_data.seek(0)
            y_data.seek(0)

            # Upload to MinIO
            DatasetManager.s3_client.put_object(
                Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/X_train.npy", Body=X_data.getvalue()
            )
            DatasetManager.s3_client.put_object(
                Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/y_train.npy", Body=y_data.getvalue()
            )
            DatasetManager.names.add(dataset_name)
            logger.success(f"Dataset '{dataset_name}' successfully saved in MinIO.")
            return dataset_name
        except Exception as e:
            logger.error(f"Error saving dataset '{dataset_name}': {e}")
            raise

    @staticmethod
    def delete_dataset(dataset_name: str) -> str:
        """
        Deletes a dataset from MinIO.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to delete.

        Returns
        -------
        str
            Name of the deleted dataset.
        """
        if dataset_name not in DatasetManager.names:
            raise ValueError(f"Dataset '{dataset_name}' does not exist in MinIO.")

        logger.info(f"Deleting dataset '{dataset_name}' from MinIO")
        try:
            DatasetManager.s3_client.delete_object(Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/X_train.npy")
            DatasetManager.s3_client.delete_object(Bucket=DatasetManager.bucket_name, Key=f"{dataset_name}/y_train.npy")
            DatasetManager.names.remove(dataset_name)
            logger.success(f"Dataset '{dataset_name}' successfully deleted from MinIO.")
            return dataset_name
        except Exception as e:
            logger.error(f"Error deleting dataset '{dataset_name}': {e}")
            raise

    @staticmethod
    def list_available() -> List[str]:
        """
        Lists all available dataset names.

        Returns
        -------
        List[str]
            A list of available dataset names.
        """
        logger.info("Listing available datasets")
        return list(DatasetManager.names)

    @staticmethod
    def list_user_defined() -> List[str]:
        """
        Lists all user defined dataset names.

        Returns
        -------
        List[str]
            A list of user defined dataset names.
        """
        logger.info("Listing user-defined datasets")
        return list(DatasetManager.names - DatasetManager.base_datasets)
