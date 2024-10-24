from fastapi import FastAPI, Query, Path, HTTPException
import uvicorn
from typing import Optional, Annotated, List
from pydantic import BaseModel, Field
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging


MODEL_PATH = "models/"

models = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Модель запроса для предсказания
class PredictionRequest(BaseModel):
    mmodel_name: str = Field(description="Имя обученной модели")
    input_data: List[List[float]] = Field(description="Входные данные для предсказания")

    class Config:
        schema_extra = {
            "example": {
                "mmodel_name": "random_forest_model.pkl",
                "input_data": [[0, 0], [1, 1], [2, 2]]
            }
        }


class ModelParams(BaseModel):
    mmodel_type: str = Field(description="Тип модели")
    params: Optional[dict] = Field(default={}, description="Гиперпараметры для модели")

    class Config:
        schema_extra = {
            "example": {
                "mmodel_type": "random_forest",
                "params": {"n_estimators": 100, "max_depth": 5}
            }
        }

        
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model Management API!"}

# Эндпоинт для обучения
@app.post("/train/", status_code=201)
async def train_model(params: ModelParams):
    try:
        mmodel_type = params.mmodel_type
        logger.info(f"Starting training for {mmodel_type}")

        if mmodel_type not in models:
            logger.error(f"Unsupported model type: {mmodel_type}")
            raise HTTPException(status_code=400, detail="Model type not supported")

        ModelClass = models[mmodel_type]
        model = ModelClass(**params.params)

        X_train = [[0, 0], [1, 1], [2, 2]]
        y_train = [0, 1, 1]
        model.fit(X_train, y_train)

        mmodel_name = f"{mmodel_type}_model.pkl"
        model_path = os.path.join(MODEL_PATH, mmodel_name)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model {mmodel_type} trained and saved as {mmodel_name}")
        return {"message": f"Model {mmodel_type} trained and saved as {mmodel_name}"}

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для переобучения
@app.post("/retrain/{mmodel_name}", status_code=200)
async def retrain_model(mmodel_name: str):
    model_path = os.path.join(MODEL_PATH, mmodel_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_train = [[1, 0], [0, 1], [2, 3]]
    y_train = [1, 0, 1]

    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {"message": f"Model {mmodel_name} retrained successfully"}

# Эндпоинт для списка доступных моделей
@app.get("/models/")
async def list_models():
    model_files = os.listdir(MODEL_PATH)
    return {"available_models": model_files}

# Эндпоинт для предсказания с обученной моделью
@app.post("/predict/", status_code=200)
async def predict(request: PredictionRequest):
    mmodel_name = request.mmodel_name
    model_path = os.path.join(MODEL_PATH, mmodel_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(request.input_data)
    return {"predictions": predictions.tolist()}

# Эндпоинт для удаления модели
@app.delete("/models/{mmodel_name}", status_code=200)
async def delete_model(
    mmodel_name: Annotated[str, Path(description="The name of the model to delete")]
):
    model_path = os.path.join(MODEL_PATH, mmodel_name)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    os.remove(model_path)
    return {"message": f"Model {mmodel_name} deleted"}

# Эндпоинт для проверки статуса сервиса
@app.get("/status/", status_code=200)
async def status_check():
    return {"status": "Service is running"}

if __name__=="__main__":
    uvicorn.run("app:app", reload=True)
