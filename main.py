from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="EW Radar Anomaly Detection API",
    description="Unsupervised ML Inference Pipeline for Electronic Warfare Radar Telemetry",
    version="1.0.0"
)

# Enable CORS for Frontend JavaScript Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Serialized ML Artifacts safely
MODEL_PATH = "radar_anomaly_model.pkl"
SCALER_PATH = "radar_scaler.pkl"

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

class TelemetryInput(BaseModel):
    frequency: float
    pulse_width: float
    pri: float
    amplitude: float

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "EW Radar Anomaly Detection REST Backend",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_anomaly(data: TelemetryInput):
    # Input vector array construction
    input_features = np.array([[
        data.frequency,
        data.pulse_width,
        data.pri,
        data.amplitude
    ]])

    # Apply scaling if scaler artifact exists
    if scaler is not None:
        input_features = scaler.transform(input_features)

    # Mock prediction logic fallback if model artifact not available locally
    if model is not None:
        prediction = model.predict(input_features)[0]
        # Isolation Forest outputs -1 for anomalies, 1 for normal data
        is_anomaly = bool(prediction == -1)
    else:
        # Standard threshold heuristic fallback
        is_anomaly = bool(data.frequency > 4000 or data.amplitude > 0)

    return {
        "is_anomaly": is_anomaly,
        "classification": "ANOMALY" if is_anomaly else "NOMINAL",
        "metrics_received": data.dict()
    }
