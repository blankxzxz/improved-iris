from contextlib import asynccontextmanager
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Global model container
# ---------------------------------------------------------
ml_models: Dict[str, Any] = {}

# ---------------------------------------------------------
# Lifespan: Load the model once on application startup
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "iris_tuned_lr.pkl"
    try:
        ml_models["iris_pipeline"] = joblib.load(model_path)
        print(f"[INFO] Loaded model successfully from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    yield
    # Cleanup when server shuts down
    ml_models.clear()
    print("[INFO] Model unloaded.")

app = FastAPI(
    title="Iris Species Classification API",
    description="Production inference endpoint for the tuned Iris Logistic Regression pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# 1. Pydantic Schemas (The Contract)
# ---------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
    }
class IrisPredictionResponse(BaseModel):
    predicted_species: str
    confidence: float
    probabilities: Dict[str, float]

# ---------------------------------------------------------
# 2. Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health", status_code=200)
async def health_check():
    if "iris_pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "healthy",
        "model_loaded": True
    }

# ---------------------------------------------------------
# 3. Prediction Endpoint
# ---------------------------------------------------------
@app.post("/predict", response_model=IrisPredictionResponse)
async def predict_species(payload: IrisInput):
    model = ml_models.get("iris_pipeline")
    if not model:
        raise HTTPException(status_code=503, detail="Model is unavailable.")

    # Match the exact column names the pipeline was trained on
    input_df = pd.DataFrame([{
        "SepalLengthCm": payload.sepal_length,
        "SepalWidthCm": payload.sepal_width,
        "PetalLengthCm": payload.petal_length,
        "PetalWidthCm": payload.petal_width
    }])

    # Generate prediction and probabilities
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    # Map class probabilities to their respective class names
    class_probs = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(model.classes_, probabilities)
    }

    return IrisPredictionResponse(
        predicted_species=prediction,
        confidence=round(float(probabilities.max()), 4),
        probabilities=class_probs
    )