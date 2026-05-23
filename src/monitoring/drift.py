"""Drift detection module using Evidently AI."""

import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class TrafficDriftDetector:
    """Detects data drift in traffic features using Evidently AI."""

    REPORT_DIR = Path("reports/drift")

    def __init__(self, report_dir: Optional[Union[str, Path]] = None):
        """Initialize the drift detector.

        Args:
            report_dir: Directory to save drift reports
        """
        self.report_dir = Path(report_dir) if report_dir else self.REPORT_DIR
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _try_import_evidently(self):
        """Try to import Evidently, raise helpful error if not available."""
        try:
            import evidently

            return evidently
        except ImportError:
            raise ImportError(
                "Evidently AI is required for drift detection. Install it with: pip install evidently"
            )

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        features: list[str],
        timestamp: Optional[datetime.datetime] = None,
    ) -> dict:
        """Detect drift between reference and current data.

        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (new data)
            features: List of features to monitor
            timestamp: Timestamp for the report

        Returns:
            Dictionary with drift detection results
        """
        self._try_import_evidently()
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        timestamp = timestamp or datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Prepare data
        reference = reference_data[features].copy()
        current = current_data[features].copy()

        numerical_features = []
        categorical_features = []
        for feat in features:
            if feat in ["hour", "day_of_week", "is_weekend", "is_holiday"]:
                categorical_features.append(feat)
            else:
                numerical_features.append(feat)

        for feat in numerical_features:
            reference[feat] = pd.to_numeric(reference[feat], errors="coerce").astype("float64")
            current[feat] = pd.to_numeric(current[feat], errors="coerce").astype("float64")

        for feat in categorical_features:
            reference[feat] = reference[feat].astype("string").fillna("missing")
            current[feat] = current[feat].astype("string").fillna("missing")

        reference = reference.replace([float("inf"), float("-inf")], float("nan")).dropna(
            subset=numerical_features
        )
        current = current.replace([float("inf"), float("-inf")], float("nan")).dropna(
            subset=numerical_features
        )

        data_definition = DataDefinition(
            numerical_columns=numerical_features,
            categorical_columns=categorical_features,
        )
        reference_dataset = Dataset.from_pandas(reference, data_definition)
        current_dataset = Dataset.from_pandas(current, data_definition)

        # Create report
        report = Report([DataDriftPreset(columns=features)])
        snapshot = report.run(current_dataset, reference_dataset, timestamp=timestamp)
        result = snapshot.dict()

        # Save report
        report_path = self.report_dir / f"drift_report_{timestamp_str}.html"
        snapshot.save_html(str(report_path))

        # Extract key metrics
        drift_metrics = {
            "timestamp": timestamp.isoformat(),
            "report_path": str(report_path),
            "n_features": len(features),
            "n_drifted_features": 0,
            "max_drift_score": 0.0,
            "drift_share": 0.0,
            "feature_drift_scores": {},
        }

        for metric in result.get("metrics", []):
            config = metric.get("config", {})
            value = metric.get("value")
            metric_type = config.get("type", "")

            if metric_type.endswith("DriftedColumnsCount") and isinstance(value, dict):
                drift_metrics["n_drifted_features"] = int(value.get("count", 0))
                drift_metrics["drift_share"] = float(value.get("share", 0.0))
                continue

            if metric_type.endswith("ValueDrift"):
                column = config.get("column")
                if column is not None and value is not None:
                    drift_score = float(value)
                    drift_metrics["feature_drift_scores"][column] = drift_score
                    drift_metrics["max_drift_score"] = max(
                        drift_metrics["max_drift_score"], drift_score
                    )

        return drift_metrics

    def monitor_feature_stats(
        self, reference_data: pd.DataFrame, current_data: pd.DataFrame, features: list[str]
    ) -> dict:
        """Monitor feature statistics for drift.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            features: List of features to monitor

        Returns:
            Dictionary with feature statistics comparison
        """
        stats = {}

        for feat in features:
            ref_stats = reference_data[feat].describe()
            curr_stats = current_data[feat].describe()

            stats[feat] = {
                "reference": {
                    "mean": ref_stats.get("mean", 0),
                    "std": ref_stats.get("std", 0),
                    "min": ref_stats.get("min", 0),
                    "max": ref_stats.get("max", 0),
                },
                "current": {
                    "mean": curr_stats.get("mean", 0),
                    "std": curr_stats.get("std", 0),
                    "min": curr_stats.get("min", 0),
                    "max": curr_stats.get("max", 0),
                },
                "diff_mean": curr_stats.get("mean", 0) - ref_stats.get("mean", 0),
                "diff_std": curr_stats.get("std", 0) - ref_stats.get("std", 0),
            }

        return stats

    def check_anomaly_rate_drift(
        self, reference_anomalies: pd.DataFrame, current_anomalies: pd.DataFrame
    ) -> dict:
        """Check if anomaly rate has drifted.

        Args:
            reference_anomalies: DataFrame with anomaly predictions for reference
            current_anomalies: DataFrame with anomaly predictions for current

        Returns:
            Dictionary with anomaly rate comparison
        """
        ref_rate = (
            reference_anomalies["is_anomaly"].mean()
            if "is_anomaly" in reference_anomalies.columns
            else 0
        )
        curr_rate = (
            current_anomalies["is_anomaly"].mean()
            if "is_anomaly" in current_anomalies.columns
            else 0
        )

        return {
            "reference_anomaly_rate": float(ref_rate),
            "current_anomaly_rate": float(curr_rate),
            "diff": float(curr_rate - ref_rate),
            "relative_change": float((curr_rate - ref_rate) / ref_rate) if ref_rate > 0 else 0.0,
        }
