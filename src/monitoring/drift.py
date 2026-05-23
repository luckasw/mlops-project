"""Drift detection module using Evidently AI."""

import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
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
                "Evidently AI is required for drift detection. Install it with: uv add evidently"
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
        evidently = self._try_import_evidently()
        from evidently import Calculator, ColumnMapping
        from evidently.calculations.data_drift import DataDriftCalculator
        from evidently.metrics import *
        from evidently.report import Report

        timestamp = timestamp or datetime.datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

        # Prepare data
        reference = reference_data[features].copy()
        current = current_data[features].copy()

        # Create column mapping
        column_mapping = ColumnMapping()
        for feat in features:
            if feat in ["hour", "day_of_week"]:
                column_mapping.categorical_features = [feat]
            elif feat in ["is_weekend", "is_holiday"]:
                column_mapping.categorical_features.append(feat)
            else:
                column_mapping.numerical_features = column_mapping.numerical_features or []
                column_mapping.numerical_features.append(feat)

        # Create report
        report = Report(
            metrics=[
                DataDriftCalculator(),
            ]
        )

        report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

        result = report.as_dict()

        # Save report
        report_path = self.report_dir / f"drift_report_{timestamp_str}.html"
        report.save_html(report_path)

        # Extract key metrics
        drift_metrics = {
            "timestamp": timestamp.isoformat(),
            "report_path": str(report_path),
            "n_features": len(features),
            "n_drifted_features": 0,
            "max_drift_score": 0.0,
        }

        # Try to extract drift information
        try:
            if result and "metrics" in result:
                for metric in result["metrics"]:
                    if metric.get("metric_name") == "DataDriftCalculator":
                        data = metric.get("result", {})
                        drift_metrics["n_drifted_features"] = data.get(
                            "number_of_drifted_features", 0
                        )
                        drift_metrics["max_drift_score"] = data.get("max_drift_score", 0.0)
        except Exception as e:
            print(f"Error extracting drift metrics: {e}")

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
