from argparse import Namespace
from time import sleep
from typing import Any, Optional, Tuple, Callable, List

import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from loguru import logger

from mops.frontend.managers.client import Client
from mops.frontend.managers.display import DisplayManager

logger = logger.bind(logger_name="frontend")


class SafeCall:
    def __init__(self, num_args: int = 1, err_holder: Optional[DeltaGenerator] = None):
        self.num_args = num_args
        self.err_holder = err_holder

    def __enter__(self) -> Callable:
        return self.call

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return True

    def call(self, method: Callable, *args, **kwargs) -> Optional[Any]:
        method_name = method.__name__

        logger.info(f"Safe calling `{method_name}`")

        try:
            result = method(*args, **kwargs)
            logger.success(f"Safe calling `{method_name}` executed successfully")
            return result
        except Exception as e:
            log_msg = f"Safe calling `{method_name}` failed: {e}"
            st_msg = f"`{method_name}` failed: {e}"

            if self.err_holder is not None:
                self.err_holder.error(st_msg)
            logger.warning(log_msg)

            if self.num_args > 1:
                return (None,) * self.num_args
            return None


class DashboardManager:
    ns = Namespace(
        datasets=None,
        models=None,
        trained_models=None,
    )

    user_ds: List[str] = []

    def __init__(self):
        pass

    @staticmethod
    def update_namespace() -> None:
        """
        Fetches available datasets, models, and trained models from the server
        and updates the namespace.
        """
        logger.info("Updating namespace")
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            DashboardManager.ns.datasets = safe_call(Client.get_available_datasets)
            DashboardManager.ns.models = safe_call(Client.get_available_models)
            DashboardManager.ns.trained_models = safe_call(Client.get_trained_models)
        logger.success("Namespace updated")

    @staticmethod
    def update_user_ds() -> None:
        """
        Fetches and updates user defined datasets.
        """
        logger.info("Updating user datasets")
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            DashboardManager.user_ds = safe_call(Client.get_user_datasets)
        logger.success("User datasets updated")

    @staticmethod
    def create_dataset(dataset_name: str, X: List[List[float]], y: List[int]) -> str:
        """
        Handles creation of dataset.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset to delete.
        X: List[List[float]]
            Training data
        y: List[int]
            Training target
        Returns:
        dataset_name: str
            The name of the created dataset.
        """
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            response = safe_call(Client.create_dataset, dataset_name, X, y)
        DashboardManager.update_namespace()
        return response

    @staticmethod
    def handle_dataset_deletion(dataset_name: str) -> str:
        """
        Handles deletion of dataset by name.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset to delete.

        Returns:
        dataset_name: str
            The name of the deleted dataset.
        """
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            response = safe_call(Client.delete_dataset, dataset_name)
        DashboardManager.update_namespace()
        DashboardManager.update_user_ds()
        return response

    @staticmethod
    def show_dataset(dataset_name: str, display: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Fetches a dataset from the server and optionally displays it.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to fetch.
        display : bool, optional
            Whether to display the dataset after fetching (default is False).

        Returns
        -------
        Optional[Tuple[np.ndarray, np.ndarray]]
            The dataset's features (X) and labels (y), or None if the fetch fails.
        """
        err_holder = st.empty()
        with SafeCall(2, err_holder) as safe_call:
            X, y = safe_call(Client.get_dataset, dataset_name)
        if X is None and y is None:
            return X, y

        if display:
            fig = DisplayManager.display_dataset(X, y)
            st.pyplot(fig)

        return X, y

    @staticmethod
    def show_docstring(model_name: str) -> Optional[str]:
        """
        Fetches the docstring from the server and displays it.

        Parameters
        ----------
        model_name : str
            The name of the model.

        Returns
        -------
        Optional[str]
            The fetched docstring, or None if the fetch fails.
        """
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            docstring = safe_call(Client.get_docstring, model_name)

        if docstring:
            DisplayManager.display_docstring(docstring)

        return docstring

    @staticmethod
    def handle_training(dataset_name: str, model_name: str, parameters: str) -> Optional[str]:
        """
        Handles the complete model training process, including fetching the dataset,
        creating a meshgrid, training the model, and displaying the results.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to use for training.
        model_name : str
            The name of the model to train.
        parameters : str
            The parameters to use for model training in json format.

        Returns
        -------
        Optional[str]
            The ID of the trained model, or None if the process fails.
        """
        try:
            parameters = eval(parameters)
        except Exception:
            err_msg = "Can't parse parameters. The parameters are probably not in JSON format"
            st.error(err_msg)
            logger.error(err_msg)
            return None

        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            model_id = safe_call(
                Client.train_model, dataset_name=dataset_name, model_name=model_name, parameters=parameters
            )
        if model_id is None:
            return model_id

        DashboardManager.update_namespace()

        scs_msg = f"Model trained and saved with ID: {model_id}"
        st.success(scs_msg)
        logger.info(scs_msg)

        with SafeCall(2, err_holder) as safe_call:
            X, y = safe_call(Client.get_dataset, dataset_name)
        if X is None or y is None:
            return model_id

        xx0, xx1 = DisplayManager.meshgrid(X)
        XX = DisplayManager.meshgrid_for_predict(X)

        logger.info(str(type(XX)))
        with SafeCall(err_holder=err_holder) as safe_call:
            yy = safe_call(Client.get_predict_proba, model_id=model_id, X=XX.tolist())
        if yy is None:
            return model_id

        fig = DisplayManager.display_train(X, y, xx0, xx1, yy)
        st.pyplot(fig)
        return model_id

    @staticmethod
    def handle_prediction(model_id: str, X_test: List[List[float]]) -> Optional[np.ndarray]:
        """
        Handles the complete prediction process, including fetching the training dataset,
        generating a meshgrid, fetching predicted probabilities, and displaying the predictions.

        Parameters
        ----------
        model_id : str
            The ID of the model to use for prediction.
        X_test : List[List[float]]
            The input dataset.

        Returns
        -------
        Optional[np.array]
            The predicted labels for the test dataset, or None if the process fails.
        """
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            y_test = safe_call(Client.get_predict, model_id=model_id, X=X_test)
        if y_test is None:
            return y_test
        st.success(f"Predictions: {y_test}")

        with SafeCall(err_holder=err_holder) as safe_call:
            train_ds_name = safe_call(Client.get_model_dataset, model_id)
        if train_ds_name is None:
            return y_test

        with SafeCall(2, err_holder) as safe_call:
            X_train, y_train = safe_call(Client.get_dataset, train_ds_name)

        if X_train is None or y_train is None:
            return y_test

        xx0, xx1 = DisplayManager.meshgrid(X_train)
        XX = DisplayManager.meshgrid_for_predict(X_train)

        with SafeCall(err_holder=err_holder) as safe_call:
            yy = safe_call(Client.get_predict_proba, model_id=model_id, X=XX.tolist())
        if yy is None:
            return y_test

        logger.info(str(type(y_test)))
        fig = DisplayManager.display_predict(X_train, y_train, xx0, xx1, yy, np.array(X_test), y_test)
        st.pyplot(fig)
        return y_test

    @staticmethod
    def handle_model_deletion(model_id: str) -> Optional[str]:
        """
        Deletes a model by its ID.

        Parameters
        ----------
        model_id : str
            The ID of the model to delete.

        Returns
        -------
        Optional[str]
            The ID of the deleted model, or None if the deletion fails.
        """
        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            model_id = safe_call(Client.delete_model, model_id)
        if model_id is None:
            return model_id

        DashboardManager.update_namespace()
        st.success(f"Model '{model_id}' deleted successfully")
        logger.success(f"Model '{model_id}' deleted successfully")
        return model_id

    @staticmethod
    def handle_retraining(model_id: str, dataset_name: str, parameters: str) -> Optional[str]:
        """
        Handles the complete retraining process, including fetching the dataset,
        retraining the model, creating a meshgrid, and displaying the results.

        Parameters
        ----------
        model_id : str
            The ID of the model to retrain.
        dataset_name : str
            The name of the dataset to use for retraining.
        parameters : str
            The parameters to use for model retraining in json format.

        Returns
        -------
        Optional[str]
            The ID of the retrained model, or None if the process fails.
        """
        try:
            parameters = eval(parameters)
        except SyntaxError:
            st.error("Can't parse parameters, make sure the parameters are inside `{}`")
            logger.error("SyntaxError. The parameters are probably not inside  `{}`")
            return None

        err_holder = st.empty()
        with SafeCall(err_holder=err_holder) as safe_call:
            model_id = safe_call(
                Client.retrain_model, model_id=model_id, dataset_name=dataset_name, parameters=parameters
            )
        if model_id is None:
            st.error("Failed to retrain a model")
            return model_id

        DashboardManager.update_namespace()

        with SafeCall(2, err_holder) as safe_call:
            X, y = safe_call(Client.get_dataset, dataset_name)
        if X is None or y is None:
            return model_id

        xx0, xx1 = DisplayManager.meshgrid(X)
        XX = DisplayManager.meshgrid_for_predict(X)

        with SafeCall(err_holder=err_holder) as safe_call:
            yy = safe_call(Client.get_predict_proba, model_id=model_id, X=XX.tolist())
        if yy is None:
            return model_id

        fig = DisplayManager.display_train(X, y, xx0, xx1, yy)
        st.pyplot(fig)
        return model_id

    @staticmethod
    def show_health(holders: List[DeltaGenerator], retry_interval: int = 5) -> None:
        """
        Continuously checks the health status of the server, retries if unhealthy.

        Parameters
        ----------
        holders: List[DeltaGenerator]
            List of holders for values. Result of st.empty().
        retry_interval : int, optional
            The number of seconds to wait before retrying (default is 5).
        """

        retry_text = st.empty()
        err_holder = st.empty()
        while True:
            try:
                with SafeCall(err_holder=err_holder) as safe_call:
                    safe_call(Client.connect)
                with SafeCall(err_holder=err_holder) as safe_call:
                    status = safe_call(Client.get_health)

                DisplayManager.display_health(status, holders)

                if status is not None:
                    err_holder.empty()
                    retry_text.empty()
                    DashboardManager.update_namespace()
                    DashboardManager.update_user_ds()
                    return

                for i in range(retry_interval, 0, -1):
                    DisplayManager.display_retry(status=status, seconds=i, retry_text=retry_text)
                    logger.info(f"Retrying connection in {i} seconds")
                    sleep(1)

            except KeyboardInterrupt:
                logger.warning("Interrupting health status check")
                return
