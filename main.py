"""Main entry point for the traffic anomaly detection MLOps pipeline."""

import fire


def train():
    """Train the anomaly detection model."""
    from src.flows.training import create_training_flow

    # Run training flow
    model_path = create_training_flow()
    print(f"Model trained and saved to: {model_path}")


def predict():
    """Run the prediction API."""
    import uvicorn

    from src.api.app import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


def dashboard():
    """Run the Streamlit dashboard."""
    import streamlit as st

    from src.dashboard.app import run_dashboard

    run_dashboard()


def retrain():
    """Run hourly retraining flow."""
    from src.flows.training import create_hourly_retraining_flow

    model_path = create_hourly_retraining_flow()
    print(f"Model retrained and saved to: {model_path}")


def monitor():
    """Run drift monitoring."""
    import datetime
    import json
    from pathlib import Path

    from src.data.loader import TrafficDataLoader
    from src.monitoring.drift import TrafficDriftDetector

    loader = TrafficDataLoader()
    processed_path = Path("data/processed/traffic_data_processed.parquet")
    if processed_path.exists():
        df = loader.load_parquet(processed_path)
    else:
        from src.data.preprocessor import TrafficDataPreprocessor
        from src.features.engineer import TrafficFeatureEngineer

        raw_df = loader.load_years([2018, 2019, 2020, 2021, 2022, 2023, 2025])
        preprocessor = TrafficDataPreprocessor()
        engineer = TrafficFeatureEngineer()
        df = engineer.engineer_features(preprocessor.preprocess(raw_df))

    # Compare historical reference data against current validation-period data.
    ref_df = df[df["year"].isin([2018, 2019, 2020, 2021, 2022, 2023])].copy()
    curr_df = df[df["year"].isin([2025])].copy()

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

    # Detect drift
    detector = TrafficDriftDetector()
    drift_results = detector.detect_drift(ref_df, curr_df, features)

    print("Drift Detection Results:")
    for key, value in drift_results.items():
        print(f"  {key}: {value}")

    metrics_path = Path("metrics.json")
    metrics_path.write_text(json.dumps(drift_results, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {metrics_path}")


def preprocess():
    """Preprocess and save data to parquet."""
    from pathlib import Path

    from src.data.loader import TrafficDataLoader
    from src.data.preprocessor import TrafficDataPreprocessor
    from src.features.engineer import TrafficFeatureEngineer

    loader = TrafficDataLoader()
    preprocessor = TrafficDataPreprocessor()
    engineer = TrafficFeatureEngineer()

    # Load all data
    df = loader.load_all()
    print(f"Loaded {len(df)} rows")

    # Preprocess
    df = preprocessor.preprocess(df)
    print(f"Preprocessed: {df.shape}")

    # Engineer features
    df = engineer.engineer_features(df)
    print(f"Engineered features: {df.shape}")

    # Save to parquet
    output_path = Path("data/processed/traffic_data_processed.parquet")
    loader.to_parquet(df, output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "predict": predict,
            "dashboard": dashboard,
            "retrain": retrain,
            "monitor": monitor,
            "preprocess": preprocess,
        }
    )
