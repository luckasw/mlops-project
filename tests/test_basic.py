#!/usr/bin/env python3
"""Basic test to verify the implementation works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.data.loader import TrafficDataLoader

        print("✓ Data loader imported")
    except ImportError as e:
        print(f"✗ Data loader import failed: {e}")
        return False

    try:
        from src.data.preprocessor import TrafficDataPreprocessor

        print("✓ Data preprocessor imported")
    except ImportError as e:
        print(f"✗ Data preprocessor import failed: {e}")
        return False

    try:
        from src.features.engineer import TrafficFeatureEngineer

        print("✓ Feature engineer imported")
    except ImportError as e:
        print(f"✗ Feature engineer import failed: {e}")
        return False

    try:
        from src.models.anomaly import TrafficAnomalyDetector

        print("✓ Anomaly detector imported")
    except ImportError as e:
        print(f"✗ Anomaly detector import failed: {e}")
        return False

    return True


def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")

    try:
        from src.data.loader import TrafficDataLoader

        loader = TrafficDataLoader()

        # Test loading a single year
        df = loader.load_year(2018)
        print(f"✓ Loaded {len(df)} rows for 2018")

        # Check columns
        expected_cols = ["id", "kanal", "aeg", "year"] + [str(i) for i in range(1, 11)]
        for col in expected_cols:
            if col not in df.columns:
                print(f"✗ Missing column: {col}")
                return False

        print("✓ All expected columns present")
        return True

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_preprocessing():
    """Test data preprocessing."""
    print("\nTesting preprocessing...")

    try:
        from src.data.loader import TrafficDataLoader
        from src.data.preprocessor import TrafficDataPreprocessor

        loader = TrafficDataLoader()
        df = loader.load_year(2018)

        preprocessor = TrafficDataPreprocessor()
        df = preprocessor.preprocess(df)

        # Check new columns
        expected_new_cols = [
            "total_vehicles",
            "avg_speed",
            "pct_heavy_vehicles",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_holiday",
        ]
        for col in expected_new_cols:
            if col not in df.columns:
                print(f"✗ Missing preprocessed column: {col}")
                return False

        print(f"✓ Preprocessing created {len(expected_new_cols)} new columns")
        return True

    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False


def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")

    try:
        from src.data.loader import TrafficDataLoader
        from src.data.preprocessor import TrafficDataPreprocessor
        from src.features.engineer import TrafficFeatureEngineer

        loader = TrafficDataLoader()
        df = loader.load_year(2018)

        preprocessor = TrafficDataPreprocessor()
        df = preprocessor.preprocess(df)

        engineer = TrafficFeatureEngineer()
        df = engineer.engineer_features(df)

        # Check feature columns
        expected_feature_cols = [
            "total_vehicles",
            "avg_speed",
            "pct_heavy_vehicles",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "rolling_avg_24h",
        ]
        for col in expected_feature_cols:
            if col not in df.columns:
                print(f"✗ Missing feature column: {col}")
                return False

        print(f"✓ Feature engineering created all {len(expected_feature_cols)} features")
        return True

    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False


def test_model_training():
    """Test model training."""
    print("\nTesting model training...")

    try:
        import tempfile
        from pathlib import Path

        from src.data.loader import TrafficDataLoader
        from src.data.preprocessor import TrafficDataPreprocessor
        from src.features.engineer import TrafficFeatureEngineer
        from src.models.anomaly import TrafficAnomalyDetector

        # Load and prepare data
        loader = TrafficDataLoader()
        df = loader.load_years([2018, 2019])

        preprocessor = TrafficDataPreprocessor()
        df = preprocessor.preprocess(df)

        engineer = TrafficFeatureEngineer()
        df = engineer.engineer_features(df)

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
        detector = TrafficAnomalyDetector(n_estimators=10, contamination=0.01)
        detector.fit(df[features], features)

        print("✓ Model trained successfully")

        # Test prediction
        predictions = detector.predict(df[features][:10])
        print(f"✓ Made {len(predictions)} predictions")

        # Test saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            detector.save(model_path)
            print(f"✓ Model saved to {model_path}")

            # Test loading
            new_detector = TrafficAnomalyDetector()
            new_detector.load(model_path)
            print("✓ Model loaded successfully")

        return True

    except Exception as e:
        print(f"✗ Model training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Traffic Anomaly Detection - Basic Tests")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Feature Engineering", test_feature_engineering()))
    results.append(("Model Training", test_model_training()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)
    print("=" * 60)

    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
