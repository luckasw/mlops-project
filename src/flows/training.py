"""Prefect flow for model training and retraining."""

import datetime
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from prefect import flow, get_run_logger, task


@task
def load_data(years: list[int]) -> pd.DataFrame:
    """Load traffic data for specified years."""
    from src.data.loader import TrafficDataLoader
    from src.data.preprocessor import TrafficDataPreprocessor

    logger = get_run_logger()
    logger.info(f"Loading data for years: {years}")

    loader = TrafficDataLoader()
    df = loader.load_years(years)

    preprocessor = TrafficDataPreprocessor(impute_speed=True)
    df = preprocessor.preprocess(df)

    logger.info(f"Loaded {len(df)} rows")
    return df


@task
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from traffic data."""
    from src.features.engineer import TrafficFeatureEngineer

    logger = get_run_logger()
    logger.info("Engineering features...")

    engineer = TrafficFeatureEngineer()
    df = engineer.engineer_features(df)

    logger.info(f"Feature matrix shape: {df.shape}")
    return df


@task
def train_model(
    X: pd.DataFrame, features: list[str], n_estimators: int = 100, contamination: float = 0.01
) -> tuple:
    """Train anomaly detection model."""
    from src.models.anomaly import TrafficAnomalyDetector

    logger = get_run_logger()
    logger.info("Training model...")

    detector = TrafficAnomalyDetector(n_estimators=n_estimators, contamination=contamination)
    detector.fit(X, features)

    logger.info("Model training complete")
    return detector, features


@task
def evaluate_model(detector: tuple, X_val: pd.DataFrame) -> dict:
    """Evaluate model on validation data."""
    detector_obj, features = detector

    logger = get_run_logger()
    logger.info("Evaluating model...")

    metrics = detector_obj.evaluate(X_val[features])

    logger.info(f"Metrics: {metrics}")
    return metrics


@task
def save_model(detector: tuple, model_path: Path) -> Path:
    """Save trained model to disk."""
    detector_obj, _ = detector

    logger = get_run_logger()
    logger.info(f"Saving model to {model_path}")

    detector_obj.save(model_path)

    return model_path


@task
def log_to_mlflow(detector: tuple, metrics: dict, model_path: Path) -> None:
    """Log model and metrics to MLflow."""
    detector_obj, features = detector

    logger = get_run_logger()
    logger.info("Logging to MLflow...")

    mlflow.set_experiment("traffic_anomaly_detection")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", detector_obj.n_estimators)
        mlflow.log_param("contamination", detector_obj.contamination)
        mlflow.log_param("max_samples", detector_obj.max_samples)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # Log model
        mlflow.sklearn.log_model(detector_obj.model, "model")

        # Log artifacts
        mlflow.log_artifact(model_path)

        # Log feature list
        with open("features.txt", "w") as f:
            f.write("\n".join(features))
        mlflow.log_artifact("features.txt")

    logger.info("MLflow logging complete")


@flow(name="train_traffic_model")
def create_training_flow(
    train_years: list[int] = [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    val_years: list[int] = [2025],
    model_path: Path = Path("models/isolation_forest.pkl"),
    n_estimators: int = 100,
    contamination: float = 0.01,
) -> Path:
    """Create a training flow for traffic anomaly detection.

    Args:
        train_years: Years for training data
        val_years: Years for validation data
        model_path: Path to save trained model
        n_estimators: Number of trees in Isolation Forest
        contamination: Expected proportion of anomalies

    Returns:
        Path to saved model
    """
    # Load and preprocess training data
    train_df = load_data(train_years)
    train_df = engineer_features(train_df)

    # Load and preprocess validation data
    val_df = load_data(val_years)
    val_df = engineer_features(val_df)

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
    detector = train_model(train_df, features, n_estimators, contamination)

    # Evaluate on validation data
    metrics = evaluate_model(detector, val_df)

    # Save model
    model_path = save_model(detector, model_path)

    # Log to MLflow
    log_to_mlflow(detector, metrics, model_path)

    return model_path


@flow(name="hourly_retraining")
def create_hourly_retraining_flow(
    new_data_year: int = 2025,
    model_path: Path = Path("models/isolation_forest.pkl"),
    n_estimators: int = 100,
    contamination: float = 0.01,
) -> Path:
    """Create a flow for hourly retraining with new data.

    Args:
        new_data_year: Year to add new data from
        model_path: Path to save/load model
        n_estimators: Number of trees in Isolation Forest
        contamination: Expected proportion of anomalies

    Returns:
        Path to saved model
    """
    # Load existing data
    historical_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    historical_df = load_data(historical_years)
    historical_df = engineer_features(historical_df)

    # Load new data
    new_df = load_data([new_data_year])
    new_df = engineer_features(new_df)

    # Combine all data
    all_df = pd.concat([historical_df, new_df], ignore_index=True)

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

    # Train model on all data
    detector = train_model(all_df, features, n_estimators, contamination)

    # Save model
    model_path = save_model(detector, model_path)

    # Log to MLflow
    metrics = detector[0].evaluate(all_df[features])
    log_to_mlflow(detector, metrics, model_path)

    return model_path
