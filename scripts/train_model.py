#!/usr/bin/env python3
"""Simple training script without Prefect dependencies."""

from pathlib import Path

from src.data.loader import TrafficDataLoader
from src.data.preprocessor import TrafficDataPreprocessor
from src.features.engineer import TrafficFeatureEngineer
from src.models.anomaly import TrafficAnomalyDetector


def train_and_save():
    """Train model and save to disk."""
    print("Loading data...")

    # Load training data (2018-2024)
    loader = TrafficDataLoader()
    train_df = loader.load_years([2018, 2019, 2020, 2021, 2022, 2023, 2024])
    print(f"Loaded {len(train_df)} training rows")

    # Preprocess
    preprocessor = TrafficDataPreprocessor()
    train_df = preprocessor.preprocess(train_df)
    print(f"Preprocessed training data")

    # Engineer features
    engineer = TrafficFeatureEngineer()
    train_df = engineer.engineer_features(train_df)
    print(f"Engineered features")

    # Define features
    features = [
        "total_vehicles",
        "avg_speed",
        "pct_heavy_vehicles",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "rolling_avg_24h",
    ]

    # Train model
    print("Training Isolation Forest model...")
    detector = TrafficAnomalyDetector(n_estimators=100, contamination=0.01, random_state=42)
    detector.fit(train_df[features], features)
    print("Model trained!")

    # Save model
    model_path = Path("models/isolation_forest.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate on validation data (2025)
    print("\nEvaluating on 2025 validation data...")
    val_df = loader.load_year(2025)
    val_df = preprocessor.preprocess(val_df)
    val_df = engineer.engineer_features(val_df)

    metrics = detector.evaluate(val_df[features])
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return model_path


if __name__ == "__main__":
    train_and_save()
