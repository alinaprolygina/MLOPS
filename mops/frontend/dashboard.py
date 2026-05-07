from argparse import Namespace

import streamlit as st
from loguru import logger

from mops.frontend.managers.dashboard import DashboardManager

logger = logger.bind(logger_name="frontend")

DATA = Namespace()
TRAIN = Namespace()
PREDICT = Namespace()
DELETE = Namespace()
RETRAIN = Namespace()


logger.info("Dashboard started")

st.title("ML Model Management Dashboard")


##################################################### HEALTH CHECK #####################################################
logger.info("Displaying health status")
columns = st.columns(5)
metric_holders = []
for col in columns:
    with col:
        holder = st.empty()
        metric_holders.append(holder)

DashboardManager.show_health(metric_holders)


####################################################### DATA MANAGING ##################################################
st.header("Data Managing")

DATA.X = st.text_area(
    "Enter trainin data (list of lists with coma separated float values)", "[[-1, -1], [0, 0], [1, 1]]"
)
DATA.y = st.text_area("Enter trainig target (list with coma separated integer values)", "[0, 1, 0]")
DATA.name = st.text_area("Enter your dataset name", "my_dataset")
if st.button("Create dataset"):
    DATA.new_name = DashboardManager.create_dataset(DATA.name, eval(DATA.X), eval(DATA.y))
    if DATA.new_name is not None:
        st.success(f"Dataset {DATA.new_name} has been created.")

        logger.info(
            "Section '{section}': Dataset created: '{dataset_name}'.",
            section="DATA",
            dataset_name=DATA.new_name,
        )
    else:
        st.error("Failed to create dataset.")

if DashboardManager.user_ds is not None:
    DATA.delete_dataset = st.selectbox("Choose a dataset to delete", DashboardManager.user_ds)
    if st.button("Delete dataset"):
        DATA.delete_name = DashboardManager.handle_dataset_deletion(DATA.delete_dataset)
        if DATA.delete_name is not None:
            st.success(f"Dataset {DATA.delete_name} has been deleted.")

            logger.info(
                "Section '{section}': Dataset deleted: '{dataset_name}'.",
                section="DATA",
                dataset_name=DATA.delete_name,
            )
        else:
            st.error("You cannot delete chosen dataset.")
else:
    st.error("Failed to fetch datasets.")
    logger.error("Section '{section}': Failed to fetch datasets.", section="Data")


####################################################### TRAINING #######################################################
st.header("Train")
# Show avaialbe datasets, display chosen
if DashboardManager.ns.datasets is not None:
    logger.info("Section '{section}': Datasets fetched successfylly", section="Train")
    TRAIN.dataset_name = st.selectbox("Choose a dataset", DashboardManager.ns.datasets)
    logger.info(
        "Section '{section}': Train dataset selected: '{dataset_name}'",
        section="Train",
        dataset_name=TRAIN.dataset_name,
    )
    X_, y_ = DashboardManager.show_dataset(TRAIN.dataset_name, display=True)
    logger.info(
        "Section '{section}': Train dataset displayed: '{dataset_name}'",
        section="Train",
        dataset_name=TRAIN.dataset_name,
    )
else:
    st.error("Failed to fetch datasets")
    logger.error("Section '{section}': Failed to fetch datasets", section="Train")

# Show avaialbe models, display docstring for chosen
if DashboardManager.ns.models is not None:
    logger.info(
        "Section '{section}': Models fetched successfylly",
        section="Train",
    )
    TRAIN.model_name = st.selectbox("Choose a model", DashboardManager.ns.models)
    logger.info(
        "Section '{section}': Model to train selected: '{model_name}'",
        section="Train",
        model_name=TRAIN.model_name,
    )

    DashboardManager.show_docstring(TRAIN.model_name)
    logger.info(
        "Section '{section}': Docstring displayed for model: '{model_name}'",
        section="Train",
        model_name=TRAIN.model_name,
    )
else:
    st.error("Failed to fetch models")
    logger.error("Section '{section}': Failed to fetch models", section="Train")

# Set parameters
TRAIN.params = st.text_area("Train Parameters (JSON). See docstring for model in the left sidebar", "{}")
logger.info("Section '{section}': Training parameters set", section="Train")

# Train the model
if st.button("Train Model"):
    logger.info("Section '{section}': 'Train Model' button pushed", section="Train")
    if TRAIN.dataset_name is not None and TRAIN.model_name is not None:
        TRAIN.trained_id = DashboardManager.handle_training(TRAIN.dataset_name, TRAIN.model_name, TRAIN.params)
        if TRAIN.trained_id is not None:
            logger.success(
                "Section '{section}': Successfully trained model: {model_name} on dataset: {dataset_name} with parameters: {params}. Model saved with ID: {trained_id}",
                section="Train",
                model_name=TRAIN.model_name,
                dataset_name=TRAIN.dataset_name,
                params=TRAIN.params,
                trained_id=TRAIN.trained_id,
            )
        else:
            logger.error(
                "Section '{section}': Error training model: {model_name} on dataset: {dataset_name} with parameters: {params}",
                section="Train",
                model_name=TRAIN.model_name,
                dataset_name=TRAIN.dataset_name,
                params=TRAIN.params,
            )
    else:
        st.error("Please choose dataset name, model name and set parameters")
        logger.error("Section '{section}': Required data for training not provided", section="Train")


###################################################### PREDICTION ######################################################
st.header("Make Predictions")
# Show trained models and select model for prediction
if DashboardManager.ns.trained_models is not None:
    logger.info("Section '{section}': Trained models fetched successfylly", section="Make Predictions")
    PREDICT.model_id = st.selectbox("Choose a trained model to make predictions", DashboardManager.ns.trained_models)
    logger.info(
        "Section '{section}': Trained model selected: '{model_id}'",
        section="Make Predictions",
        model_id=PREDICT.model_id,
    )
else:
    st.error("Failed to fetch trained models")
    logger.error("Section '{section}': Failed to fetch trained models", section="Make Predictions")

# Input testing data
PREDICT.test_data = st.text_area("Enter test data (list of lists with coma separated values)", "[[]]")
logger.info("Section '{section}': Test data set", section="Make Predictions")

# Make predictions
if st.button("Predict"):
    logger.info("Section '{section}': 'Predict' button pushed", section="Make Predictions")
    if PREDICT.model_id is not None and PREDICT.test_data is not None:
        PREDICT.X_test = eval(PREDICT.test_data)
        PREDICT.y_test = DashboardManager.handle_prediction(PREDICT.model_id, PREDICT.X_test)
        if PREDICT.y_test is not None:
            logger.success("Section '{section}': Successfully predicted labels", section="Make Predictions")
        else:
            logger.error(
                "Section '{section}': Error predicting labels for model: {model_id} on testing points: {test_points}",
                section="Make Predictions",
                model_id=PREDICT.model_id,
                test_points=PREDICT.X_test,
            )
    else:
        st.error("Please choose model ID and provide testing points")
        logger.error("Section '{section}': Required data for prediction not provided", section="Make Predictions")


####################################################### DELETION #######################################################
st.header("Delete")
if DashboardManager.ns.trained_models is not None:
    logger.info("Section '{section}': Trained models fetched successfylly", section="Delete")
    DELETE.model_id = st.selectbox("Choose a trained model to delete", DashboardManager.ns.trained_models)
    logger.info(
        "Section '{section}': Model to delete selected: '{deleting_id}'",
        section="Delete",
        deleting_id=DELETE.model_id,
    )
else:
    st.error("Failed to fetch trained models")
    logger.error("Section '{section}': Failed to fetch trained models", section="Delete")

if st.button("Delete"):
    logger.info("Section '{section}': 'Delete' button pushed", section="Delete")
    if DELETE.model_id is not None:
        DELETE.deleted_id = DashboardManager.handle_model_deletion(DELETE.model_id)
        if DELETE.deleted_id is not None:
            logger.success(
                "Section '{section}': Successfully deleted model: {deleted_id}",
                section="Delete",
                deleted_id=DELETE.deleted_id,
            )
    else:
        st.error("Please provide a model ID to delete")
        logger.error("Section '{section}': Model ID not provided", section="Delete")


###################################################### RETRAINING ######################################################
st.header("Retrain")

# Show available models and select a model
if DashboardManager.ns.trained_models is not None:
    logger.info("Section '{section}': Trained models fetched successfylly", section="Retrain")
    RETRAIN.model_id = st.selectbox("Choose a model to retrain", DashboardManager.ns.trained_models)
    logger.info(
        "Section '{section}': Model to retrain selected: '{deleting_id}'",
        section="Retrain",
        deleting_id=RETRAIN.model_id,
    )
else:
    st.error("Failed to fetch models")
    logger.error("Section '{section}': Failed to fetch trained models", section="Retrain")

# Show available datasets and select a dataset
if DashboardManager.ns.datasets is not None:
    logger.info("Section '{section}': Datasets fetched successfylly", section="Retrain")
    RETRAIN.datasets = DashboardManager.ns.datasets
    RETRAIN.dataset_name = st.selectbox("Choose a dataset to retrain on", RETRAIN.datasets)
    logger.info(
        "Section '{section}': Train dataset selected: '{dataset_name}'",
        section="Retrain",
        dataset_name=RETRAIN.dataset_name,
    )
else:
    st.error("Failed to fetch datasets")
    logger.error("Section '{section}': Failed to fetch datasets", section="Retrain")

# Set parameters
RETRAIN.params = st.text_area("Retrain Parameters (JSON)", "{}")
logger.info("Section '{section}': Training parameters set", section="Retrain")

if st.button("Retrain Model"):
    logger.info("Section '{section}': 'Retrain Model' button pushed", section="Retrain")
    if RETRAIN.model_id is not None and RETRAIN.dataset_name is not None and RETRAIN.params is not None:
        RETRAIN.retrained_id = DashboardManager.handle_retraining(
            RETRAIN.model_id, RETRAIN.dataset_name, RETRAIN.params
        )
        if RETRAIN.retrained_id is not None:
            logger.success(
                "Section '{section}': Successfully retrained model: {model_id} on dataset: {datase_name} with parameters: {params}",
                section="Retrain",
                model_id=RETRAIN.model_id,
                datase_name=RETRAIN.dataset_name,
                params=RETRAIN.params,
                retrained_id=RETRAIN.retrained_id,
            )
        else:
            logger.error(
                "Section '{section}': Error training model: {model_id} on dataset: {dataset_name} with parameters: {params}",
                section="Retrain",
                model_id=RETRAIN.model_id,
                dataset_name=RETRAIN.dataset_name,
                params=RETRAIN.params,
            )
    else:
        st.error("Please choose dataset name, model ID and set parameters")
        logger.error("Section '{section}': Required data for retraining not provided", section="Retrain")
