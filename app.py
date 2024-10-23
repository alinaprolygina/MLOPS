from fastapi import FastAPI, Query, Path, HTTPException
import uvicorn
from typing import Optional, Annotated, List
from pydantic import BaseModel, Field
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


MODEL_PATH = "models/"

models = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Модель гиперпараметров для обучения
class ModelParams(BaseModel):
    mmodel_type: str
    params: Optional[dict] = Field(default={})

# Модель запроса для предсказания
class PredictionRequest(BaseModel):
    mmodel_name: str
    input_data: List[List[float]]


class User(BaseModel): # Модель json данных
    name : str
    age : int = 18

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Gleb",
                    "age": 25
                }
            ]
        }
    }

app = FastAPI()

# Эндпоинт для обучения
@app.post("/train/", status_code=201)
async def train_model(params: ModelParams):
    mmodel_type = params.mmodel_type

    if mmodel_type not in models:
        raise HTTPException(status_code=400, detail="Model type not supported")

    ModelClass = models[mmodel_type]

    model = ModelClass(**params.params)

    # Рандомные данные, надо поменять!
    X_train = [[0, 0], [1, 1], [2, 2]]
    y_train = [0, 1, 1]
    model.fit(X_train, y_train)

    mmodel_name = f"{mmodel_type}_model.pkl"
    model_path = os.path.join(MODEL_PATH, mmodel_name)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {"message": f"Model {mmodel_type} trained and saved as {mmodel_name}"}

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
