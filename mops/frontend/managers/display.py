from typing import Tuple, Dict, Any, List

from humanize import naturalsize
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from matplotlib import figure
from loguru import logger

logger = logger.bind(logger_name="frontend")


class DisplayManager:
    cmap = plt.cm.coolwarm
    test_cmap = plt.cm.bwr
    colors = [cmap(0), cmap(255)]
    test_colors = [test_cmap(0), test_cmap(255)]

    retry_text_html = "<p style='text-align: center;'>Retrying connection in {seconds} seconds...</p>"

    def __init__(self):
        pass

    @staticmethod
    def display_dataset(X: np.ndarray, y: np.ndarray) -> figure.Figure:
        """
        Displays a scatter plot of the dataset.

        Parameters
        ----------
        X : np.ndarray
            Features of the dataset.
        y : np.ndarray
            Labels of the dataset.

        Returns
        -------
        figure.Figure
            A matplotlib figure object with the dataset scatter plot.
        """
        fig, ax = plt.subplots()
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color=DisplayManager.colors[0], label="Class 0", s=50, edgecolor="k")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color=DisplayManager.colors[1], label="Class 1", s=50, edgecolor="k")
        ax.set_xlabel("x_0")
        ax.set_ylabel("x_1")
        ax.set_title("Labeled Data Points")
        ax.legend()
        return fig

    @staticmethod
    def meshgrid(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a meshgrid from the dataset for plotting decision boundaries.

        Parameters
        ----------
        X : np.ndarray
            Dataset features.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Meshgrid arrays for plotting.
        """
        x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.1), np.arange(x1_min, x1_max, 0.1))
        return xx0, xx1

    @staticmethod
    def meshgrid_for_predict(X: np.ndarray) -> np.ndarray:
        """
        Creates a meshgrid for prediction purposes.

        Parameters
        ----------
        X : np.ndarray
            Dataset features.

        Returns
        -------
        np.ndarray
            Flattened meshgrid suitable for prediction input.
        """
        xx0, xx1 = DisplayManager.meshgrid(X)
        XX = np.c_[xx0.ravel(), xx1.ravel()]
        return XX

    @staticmethod
    def display_train(X: np.ndarray, y: np.ndarray, xx0: np.ndarray, xx1: np.ndarray, yy: np.ndarray) -> figure.Figure:
        """
        Displays a scatter plot of the training data with a model probability heatmap.

        Parameters
        ----------
        X : np.ndarray
            Training dataset features.
        y : np.ndarray
            Training dataset labels.
        xx0 : np.ndarray
            Meshgrid for the first axis.
        xx1 : np.ndarray
            Meshgrid for the second axis.
        yy : np.ndarray
            Model predicted probabilities reshaped for contour plot.

        Returns
        -------
        figure.Figure
            A matplotlib figure object with the training dataset and heatmap.
        """
        yy = yy.reshape(xx0.shape)

        fig, ax = plt.subplots()
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color=DisplayManager.colors[0], label="Class 0", s=50, edgecolor="k")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color=DisplayManager.colors[1], label="Class 1", s=50, edgecolor="k")
        contour = ax.contourf(xx0, xx1, yy, alpha=0.5, cmap=DisplayManager.cmap, levels=25, vmin=0, vmax=1)
        cbar = plt.colorbar(contour)
        cbar.set_label("Predicted Probability")
        ax.set_xlabel("x_0")
        ax.set_ylabel("x_1")
        ax.set_title("Data Points with Model Probability Heatmap")
        ax.legend()
        return fig

    @staticmethod
    def display_predict(
        X: np.ndarray,
        y: np.ndarray,
        xx0: np.ndarray,
        xx1: np.ndarray,
        yy: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> figure.Figure:
        """
        Displays a scatter plot of the training and test data along with model prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Training dataset features.
        y : np.ndarray
            Training dataset labels.
        xx0 : np.ndarray
            Meshgrid for the first axis.
        xx1 : np.ndarray
            Meshgrid for the second axis.
        yy : np.ndarray
            Model predicted probabilities reshaped for contour plot.
        X_test : np.ndarray
            Test dataset features.
        y_test : np.ndarray
            Test dataset labels.

        Returns
        -------
        figure.Figure
            A matplotlib figure object with the train and test data along with prediction heatmap.
        """
        fig, ax = plt.subplots()
        ax.scatter(X[y == 0, 0], X[y == 0, 1], color=DisplayManager.colors[0], label="Train 0", s=50, edgecolor="k")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], color=DisplayManager.colors[1], label="Train 1", s=50, edgecolor="k")
        if yy is not None:
            yy = yy.reshape(xx0.shape)
            contour = ax.contourf(xx0, xx1, yy, alpha=0.5, cmap=DisplayManager.cmap, levels=25, vmin=0, vmax=1)
            cbar = plt.colorbar(contour)
            cbar.set_label("Predicted Probability")
        ax.scatter(
            X_test[y_test == 0, 0],
            X_test[y_test == 0, 1],
            color=DisplayManager.test_colors[0],
            label="Predict 0",
            s=100,
            edgecolor="k",
        )
        ax.scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            color=DisplayManager.test_colors[1],
            label="Predict 1",
            s=100,
            edgecolor="k",
        )
        ax.set_xlabel("x_0")
        ax.set_ylabel("x_1")
        ax.set_title("Train and Test Data with Model Predictions")
        ax.legend()
        return fig

    @staticmethod
    def display_docstring(docstring: str) -> None:
        """
        Displays the docstring.

        Parameters
        ----------
        docstring : str
            The docstring to display.

        Returns
        -------
        None
        """
        with st.sidebar.header("Model Docstring"):
            st.text(docstring)

        return docstring

    @staticmethod
    def display_health(
        status: Dict[str, Any],
        holders: List[DeltaGenerator],
    ) -> None:
        """
        Updates health status box.

        Parameters
        ----------
        status (Dict[str, Any]): Status of server. Dictionary with cpu and mem usage info.
        holders (List[DeltaGenerator]): List of holders for values. Result of st.empty()
        Returns
        -------
        None
        """
        if status is not None:
            cpu_usage = f"{status['cpu_usage_percent']:.2f}%"
            memory_total = naturalsize(status["memory_usage"]["total"])
            memory_used = naturalsize(status["memory_usage"]["used"])
            memory_free = naturalsize(status["memory_usage"]["free"])
            memory_percent = f"{status['memory_usage']['percent']:.2f}%"
        else:
            cpu_usage = None
            memory_total = None
            memory_used = None
            memory_free = None
            memory_percent = None

        holders[0].metric("CPU Usage", cpu_usage)
        holders[1].metric("Memory Total", memory_total)
        holders[2].metric("Memory Used", memory_used)
        holders[3].metric("Memory Free", memory_free)
        holders[4].metric("Memory Usage (%)", memory_percent)

    @staticmethod
    def display_retry(
        status: Dict[str, Any],
        seconds: int,
        retry_text: DeltaGenerator,
    ) -> None:
        """
        Updates health retry text.

        Parameters
        ----------
        status (Dict[str, Any]): Status of server. Dictionary with cpu and mem usage info.
        seconds (int): Seconds left before reconnection.
        retry_text (DeltaGenerator): Streamlit object to markdown.

        Returns
        -------
        None
        """
        if status is not None:
            retry_text.empty()
        else:
            retry_text.markdown(DisplayManager.retry_text_html.format(seconds=seconds), unsafe_allow_html=True)
