"""FastAPI microservice for traffic anomaly detection."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Load model
class TrafficAnomalyAPI:
    """API wrapper for traffic anomaly detection."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None

    def load_model(self, model_path: Path = Path("models/isolation_forest.pkl")):
        """Load trained model."""
        import joblib

        if not model_path.exists():
            print(f"Model not found: {model_path}, continuing without model")
            self.features = [
                "total_vehicles",
                "avg_speed",
                "pct_heavy_vehicles",
                "hour",
                "day_of_week",
                "is_weekend",
                "is_holiday",
                "rolling_avg_24h",
            ]
            return
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]

    def predict(self, features: dict) -> dict:
        """Predict anomaly for a single observation."""
        if self.model is None:
            # Return a dummy response if model not loaded
            return {
                "anomaly_score": 0.0,
                "prediction": 1,
                "is_anomaly": False,
                "anomaly_score_normalized": 0.0,
            }

        # Create DataFrame from features
        df = pd.DataFrame([features])

        # Ensure all required features are present
        for feat in self.features:
            if feat not in df.columns:
                raise ValueError(f"Missing feature: {feat}")

        # Scale and predict
        X_scaled = self.scaler.transform(df[self.features])
        score = self.model.score_samples(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        return {
            "anomaly_score": float(score),
            "prediction": int(prediction),
            "is_anomaly": prediction == -1,
            "anomaly_score_normalized": float((score + 0.5) / 0.5 if score <= 0 else 1.0),
        }


# Initialize API
api = TrafficAnomalyAPI()


class PredictionRequest(BaseModel):
    """Request model for anomaly prediction."""

    total_vehicles: float = Field(..., description="Total number of vehicles")
    avg_speed: float = Field(..., description="Average speed in km/h")
    pct_heavy_vehicles: float = Field(..., description="Percentage of heavy vehicles")
    hour: int = Field(..., description="Hour of day (0-23)", ge=0, le=23)
    day_of_week: int = Field(..., description="Day of week (0-6, Monday=0)", ge=0, le=6)
    is_weekend: bool = Field(..., description="Whether it's a weekend")
    is_holiday: bool = Field(..., description="Whether it's a holiday")
    rolling_avg_24h: float = Field(..., description="24-hour rolling average of vehicles")


class BatchPredictionRequest(BaseModel):
    """Request model for batch anomaly prediction."""

    observations: list[dict] = Field(..., description="List of observations to predict")


class PredictionResponse(BaseModel):
    """Response model for anomaly prediction."""

    anomaly_score: float
    prediction: int
    is_anomaly: bool
    anomaly_score_normalized: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch anomaly prediction."""

    predictions: list[PredictionResponse]
    n_anomalies: int
    n_normal: int


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Traffic Anomaly Detection API",
        description="API for detecting traffic anomalies in real-time",
        version="0.1.0",
        docs_url="/",
        redoc_url=None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        try:
            api.load_model()
        except FileNotFoundError:
            # Model will be loaded on first request if not available
            pass

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_anomaly(request: PredictionRequest):
        """Predict anomaly for a single observation."""
        try:
            result = api.predict(request.model_dump())
            return PredictionResponse(**result)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(request: BatchPredictionRequest):
        """Predict anomalies for a batch of observations."""
        try:
            predictions = []
            for obs in request.observations:
                result = api.predict(obs)
                predictions.append(PredictionResponse(**result))

            n_anomalies = sum(1 for p in predictions if p.is_anomaly)
            n_normal = len(predictions) - n_anomalies

            return BatchPredictionResponse(
                predictions=predictions, n_anomalies=n_anomalies, n_normal=n_normal
            )
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model_loaded": api.model is not None}

    @app.get("/features")
    async def get_features():
        """Get list of required features."""
        return {
            "features": api.features
            or [
                "total_vehicles",
                "avg_speed",
                "pct_heavy_vehicles",
                "hour",
                "day_of_week",
                "is_weekend",
                "is_holiday",
                "rolling_avg_24h",
            ]
        }

    return app


# Create default app instance
app = create_app()
