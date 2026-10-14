from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_demand_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_train.csv"


app = FastAPI(title="AI Demand Forecasting API",
              description="An API for demand prediction using a trained LightGBM forecasting model",
              version="1.0.0")


class PredictionRequest(BaseModel):
    store: int = Field(gt=0, description="Positive store ID")
    date: str = Field(description="Prediction date in YYYY-MM-DD format")
    
class PredictionResponse(BaseModel):
    store: int
    date: str
    predicted_sales: float
    model: str
    
model = None
feature_columns: list[str] = []
model_load_error: str | None = None


def load_model() -> None:
    """ Load the saved model and identify the feature columns it expects. The API can start even if these files are missing, but prediction requests will return an error."""
    
    global model, feature_columns, model_load_error
    
    try:
        if not MODEL_PATH.exists():
            raise FileExistsError(f'Moel files not found: {MODEL_PATH}')
        
        if not FEATURES_PATH.exists():
            raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")
        
        model = joblib.load(MODEL_PATH)
        
        training_features = pd.read_csv(FEATURES_PATH, nrows=5)
        excluded_columns = {"Sales","Date"}
        feature_columns = [column for column in training_features.columns if column not in excluded_columns]
        
        if not feature_columns:
            raise ValueError("No model features columns were found")
        
        model_load_error = None
    
    
    except Exception as exct:
        model = None
        feature_columns = []
        model_load_error = str(exc)
        
        
load_model()




@app.get("/")
def root() -> dict[str, str]:
    return {
        "message" : "AI Demand Forecasting API is running.",
        "documentation": "/docs",
    }
    

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "error": model_load_error
    }
    


@app.post("/predict", response_model=PredictionResponse)

def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict sales for a store and date using a feature row from the historical dataset.
    
    Important:
    This initial endpoint only supports dates for which a matching engineered feature row already exists. It is not yet a fully future-capable forecasting API.
    
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail=(f"The forecasting model is unavailable. Reason: {model_load_error}"))
    
    try:
        prediction_date = pd.to_datetime(
            request.date,
            format="%Y-%m-%d",
            errors="raise"
        )
        
        if prediction_date.strftime("%Y-%m-%d") != request.date:
            raise ValueError("Date must use YYYY-MM-DD format.")

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid date. Use YYYY-MM-DD."
        )
        
    
    try:
        matching_rows = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
        
        matching_rows = matching_rows.loc[
            (matching_rows["Store"] == request.store) &
            (matching_rows["Date"].dt.normalize() == prediction_date.normalize())
        ]
        
    
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not rend feature dataL {exc}"
        )
    
    if matching_rows.empty:
        raise HTTPException(status_code=404,
                            detail=(
                                "No engineered features row exists for this store and date. Future-date feature generation Will be added in a later step."
                            ))
    
    try:
        missing_columns =[column for column in feature_columns if column not in matching_rows.columns]
        
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")
        
        model_input = matching_rows[feature_columns].copy()
        
        booster = model.booster_()
        categorical_names = booster.pandas_categorical
        expected_names = booster.feature_name()
        
        if all(name in model_input.columns for name in expected_names):
            model_input = model_input[expected_names]
            
        categorical_feature_names = set()
        
        
        if categorical_names:
            categorical_feature_names = {
                column for column in model_input.columns if str(model_input[column].dtype) in {"object","category"}
            }
            
        
        for column in categorical_feature_names:
            model_input[column] = model_input[column].astype("category")
            
            
        predicted_sales = float(model.predict(model_input)[0])
        prediction_sales = max(0.0, predicted_sales)
    
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    
    
    return PredictionResponse(
        store=request.store,
        date=prediction_date.strftime("%Y-%m-%d"),
        predicted_sales= round(predicted_sales, 2),
        model="LightGBM"
    )