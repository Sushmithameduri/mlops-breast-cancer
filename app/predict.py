import joblib
import numpy as np
import os

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train the model first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Scaler not found. Train the model first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def predict(features: list):
    if len(features) != 30:
        raise ValueError("Expected exactly 30 input features.")

    model, scaler = load_artifacts()

    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction = model.predict(features_scaled)[0]

    return int(prediction)