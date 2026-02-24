"""
FastAPI application for Breast Cancer Classification
Serves predictions using MLflow model and scaler
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Classification API",
    description="Machine Learning API for breast cancer diagnosis prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Load model and scaler
MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"✗ Error loading model/scaler: {e}")
    model = None
    scaler = None


class PredictionInput(BaseModel):
    """Input schema for prediction request"""
    features: List[float] = Field(
        ..., 
        min_items=30, 
        max_items=30,
        description="30 feature values from breast cancer cell measurements"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
                    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
                    0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
                    0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response"""
    prediction: int = Field(..., description="Predicted class: 0 (malignant) or 1 (benign)")
    prediction_label: str = Field(..., description="Human-readable label")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: dict = Field(..., description="Probability for each class")


@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Breast Cancer Classification API",
        "version": "1.0.0",
        "status": "running",
        "model": "BreastCancerModel v1",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail="Service unavailable - Model or scaler not loaded"
        )
    
    return {
        "status": "healthy",
        "model_type": "LogisticRegression",
        "scaler_type": "StandardScaler",
        "model_version": "v1",
        "features_required": 30
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Make a prediction on breast cancer data
    
    Returns:
    - prediction: 0 (malignant) or 1 (benign)
    - prediction_label: Human-readable label
    - confidence: Probability of predicted class
    - probabilities: Probabilities for both classes
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded"
        )
    
    try:
        # Convert input to numpy array
        features = np.array(input_data.features).reshape(1, -1)
        
        # CRITICAL: Scale features before prediction
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get confidence and labels
        confidence = float(probabilities[prediction])
        labels = {0: "malignant", 1: "benign"}
        prediction_label = labels[int(prediction)]
        
        return PredictionOutput(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities={
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded"
        )
    
    # Load metadata if available
    metadata = {}
    try:
        import json
        with open("models/metadata.json", "r") as f:
            metadata = json.load(f)
    except:
        pass
    
    return {
        "model_type": type(model).__name__,
        "scaler_type": type(scaler).__name__,
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist(),
        "class_labels": {0: "malignant", 1: "benign"},
        "metrics": metadata.get("metrics", {}),
        "hyperparameters": metadata.get("hyperparameters", {})
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)