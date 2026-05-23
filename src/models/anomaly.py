"""Anomaly detection model for traffic data."""

from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class TrafficAnomalyDetector:
    """Isolation Forest-based anomaly detector for traffic data."""

    MODEL_DIR = Path("models")
    DEFAULT_MODEL_NAME = "isolation_forest.pkl"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        max_samples: str = "auto",
        random_state: int = 42,
        model_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the anomaly detector.

        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of anomalies
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
            model_path: Path to load/save model
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_path = (
            Path(model_path) if model_path else self.MODEL_DIR / self.DEFAULT_MODEL_NAME
        )
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.features_: Optional[list] = None
        self.model_dir = self.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, X: pd.DataFrame, features: list[str]) -> "TrafficAnomalyDetector":
        """Train the anomaly detection model.

        Args:
            X: Feature matrix DataFrame
            features: List of feature column names

        Returns:
            Self for chaining
        """
        self.features_ = features

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X[features])

        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 for anomaly, 1 for normal).

        Args:
            X: Feature matrix DataFrame

        Returns:
            Array of predictions (-1 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X[self.features_])
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomaly scores (lower = more anomalous).

        Args:
            X: Feature matrix DataFrame

        Returns:
            Array of anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X[self.features_])
        return self.model.score_samples(X_scaled)

    def detect_anomalies(self, X: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """Detect anomalies and return DataFrame with results.

        Args:
            X: Feature matrix DataFrame
            threshold: Custom threshold for anomaly score. If None, uses model's threshold.

        Returns:
            DataFrame with anomaly predictions and scores
        """
        scores = self.predict_proba(X)
        predictions = self.predict(X)

        if threshold is None:
            threshold = np.percentile(scores, 100 * self.contamination)

        is_anomaly = predictions == -1

        result = X.copy()
        result["anomaly_score"] = scores
        result["anomaly_prediction"] = predictions
        result["is_anomaly"] = is_anomaly
        result["anomaly_score_normalized"] = (scores - scores.min()) / (scores.max() - scores.min())

        return result

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save model and scaler to disk.

        Args:
            path: Custom path to save. If None, uses model_path.
        """
        save_path = Path(path) if path else self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "features": self.features_,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
        }

        joblib.dump(data, save_path)

    def load(self, path: Optional[Union[str, Path]] = None) -> "TrafficAnomalyDetector":
        """Load model and scaler from disk.

        Args:
            path: Custom path to load. If None, uses model_path.

        Returns:
            Self with loaded model
        """
        load_path = Path(path) if path else self.model_path

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        data = joblib.load(load_path)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features_ = data["features"]
        self.n_estimators = data.get("n_estimators", 100)
        self.contamination = data.get("contamination", 0.01)
        self.max_samples = data.get("max_samples", "auto")
        self.random_state = data.get("random_state", 42)

        return self

    def evaluate(self, X: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> dict:
        """Evaluate model performance.

        Args:
            X: Feature matrix DataFrame
            y_true: True labels (optional)

        Returns:
            Dictionary with evaluation metrics
        """
        scores = self.predict_proba(X)
        predictions = self.predict(X)

        metrics = {
            "n_samples": len(X),
            "n_anomalies": (predictions == -1).sum(),
            "anomaly_percentage": (predictions == -1).mean() * 100,
            "score_mean": scores.mean(),
            "score_std": scores.std(),
            "score_min": scores.min(),
            "score_max": scores.max(),
        }

        return metrics
