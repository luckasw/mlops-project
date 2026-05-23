"""Feature engineering module for traffic anomaly detection."""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrafficFeatureEngineer:
    """Engineers features from preprocessed traffic data for anomaly detection."""

    VEHICLE_COLS = [str(i) for i in range(1, 11)]
    HEAVY_COLS = ["6", "7", "8"]
    SPEED_COLS = [
        "<40Kph",
        "40-<50",
        "50-<60",
        "60-<70",
        "70-<80",
        "80-<90",
        "90-<100",
        "100-<110",
        "110-<120",
        "120-<130",
        "=>130",
    ]

    FEATURE_COLS = [
        "total_vehicles",
        "avg_speed",
        "pct_heavy_vehicles",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "rolling_avg_24h",
    ]

    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = StandardScaler()

    def _calculate_rolling_avg(self, df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
        """Calculate 24-hour rolling average of total_vehicles per id.

        Args:
            df: DataFrame with traffic data
            id_col: Column name for device ID

        Returns:
            DataFrame with rolling average column
        """
        # Sort by id and timestamp for rolling calculation
        sort_cols = [id_col, "aeg"] if "kanal" not in df.columns else [id_col, "kanal", "aeg"]
        df = df.sort_values(sort_cols)

        # Calculate rolling average per id (or per id+kanal if kanal exists)
        group_cols = [id_col] if "kanal" not in df.columns else [id_col, "kanal"]
        df["rolling_avg_24h"] = df.groupby(group_cols)["total_vehicles"].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )

        return df

    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in features.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with imputed values
        """
        # Impute avg_speed with historical median per id and hour
        df["avg_speed"] = df.groupby(["id", "hour"])["avg_speed"].transform(
            lambda x: x.fillna(x.median())
        )

        # If still missing, use overall median
        df["avg_speed"] = df["avg_speed"].fillna(df["avg_speed"].median())

        # Fill other missing values
        df["rolling_avg_24h"] = df["rolling_avg_24h"].fillna(df["total_vehicles"])

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from preprocessed traffic data.

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Calculate total vehicles (if not already present)
        if "total_vehicles" not in df.columns:
            df["total_vehicles"] = df[self.VEHICLE_COLS].sum(axis=1)

        # Calculate weighted average speed (if not already present)
        if "avg_speed" not in df.columns:
            speed_midpoints = {
                "<40Kph": 20,
                "40-<50": 45,
                "50-<60": 55,
                "60-<70": 65,
                "70-<80": 75,
                "80-<90": 85,
                "90-<100": 95,
                "100-<110": 105,
                "110-<120": 115,
                "120-<130": 125,
                "=>130": 135,
            }
            speed_cols_weighted = df[self.SPEED_COLS].mul(
                [speed_midpoints[col] for col in self.SPEED_COLS], axis=1
            )
            total_speed_weight = speed_cols_weighted.sum(axis=1)
            total_speed_count = df[self.SPEED_COLS].sum(axis=1)
            df["avg_speed"] = np.where(
                total_speed_count > 0, total_speed_weight / total_speed_count, np.nan
            )

        # Calculate percentage of heavy vehicles (if not already present)
        if "pct_heavy_vehicles" not in df.columns:
            df["heavy_vehicle_sum"] = df[self.HEAVY_COLS].sum(axis=1)
            df["pct_heavy_vehicles"] = np.where(
                df["total_vehicles"] > 0, df["heavy_vehicle_sum"] / df["total_vehicles"], 0.0
            )

        # Calculate rolling average
        df = self._calculate_rolling_avg(df)

        # Impute missing values
        df = self._impute_missing_values(df)

        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get feature matrix for modeling.

        Args:
            df: DataFrame with engineered features

        Returns:
            DataFrame with only feature columns
        """
        return df[self.FEATURE_COLS].copy()

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features for modeling.

        Args:
            df: DataFrame with features
            fit: Whether to fit the scaler or just transform

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if fit:
            features = df[self.FEATURE_COLS].values
            scaled_features = self.scaler.fit_transform(features)
        else:
            features = df[self.FEATURE_COLS].values
            scaled_features = self.scaler.transform(features)

        scaled_df = pd.DataFrame(
            scaled_features, columns=[f"{col}_scaled" for col in self.FEATURE_COLS], index=df.index
        )

        return pd.concat([df, scaled_df], axis=1)
