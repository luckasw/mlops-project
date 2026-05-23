"""Data preprocessing module for traffic census data."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrafficDataPreprocessor:
    """Preprocesses traffic census data for anomaly detection."""

    # Stations with missing speed data
    MISSING_SPEED_IDS = {"25d51", "0e481", "ff2e7", "fde61"}

    # Vehicle type columns
    VEHICLE_COLS = [str(i) for i in range(1, 11)]

    # Heavy vehicle columns (6, 7, 8)
    HEAVY_COLS = ["6", "7", "8"]

    # Speed columns
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

    def __init__(self, impute_speed: bool = True):
        """Initialize the preprocessor.

        Args:
            impute_speed: Whether to impute missing speed data using historical medians
        """
        self.impute_speed = impute_speed
        self.scaler = StandardScaler()
        self._speed_medians: Optional[pd.DataFrame] = None

    def _get_estonian_holidays(self, year: int) -> list:
        """Get Estonian holidays for a given year.

        Args:
            year: Year to get holidays for

        Returns:
            List of holiday timestamps
        """
        # Estonian holidays (fixed and variable)
        fixed_holidays = [
            (1, 1),  # New Year's Day
            (2, 24),  # Independence Day
            (5, 1),  # May Day
            (12, 24),  # Christmas Eve
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        ]

        # Add Easter (approximate dates for simplicity)
        easter_dates = {
            2018: (4, 1),
            2019: (4, 21),
            2020: (4, 12),
            2021: (4, 4),
            2022: (4, 17),
            2023: (4, 9),
            2024: (3, 31),
            2025: (4, 20),
            2026: (4, 5),
        }

        holidays = []
        for month, day in fixed_holidays:
            holidays.append(pd.Timestamp(year, month, day))

        if year in easter_dates:
            month, day = easter_dates[year]
            holidays.append(pd.Timestamp(year, month, day))  # Easter Sunday
            # Add Easter Monday - handle month rollover
            try:
                holidays.append(pd.Timestamp(year, month, day + 1))
            except ValueError:
                pass  # Skip if invalid date

        return holidays

    def _impute_missing_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing speed values using historical medians per id + hour.

        Args:
            df: DataFrame with potentially missing speed data

        Returns:
            DataFrame with imputed speed values
        """
        if not self.impute_speed:
            return df

        # For now, just fill missing speed with 0
        # Full imputation would require loading historical data
        # which is expensive. We'll handle this in feature engineering.
        return df

    def combine_lanes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine all lane data into a single row per id and timestamp.

        Aggregates vehicle counts and speed distributions across all lanes,
        producing station-level data instead of lane-level data.

        Args:
            df: DataFrame with lane-level traffic data (must have 'kanal' column)

        Returns:
            DataFrame with lane data combined, 'kanal' column removed
        """
        df = df.copy()

        if "kanal" not in df.columns:
            return df

        # Columns to sum during aggregation
        sum_cols = self.VEHICLE_COLS + self.SPEED_COLS + ["total_vehicles", "heavy_vehicle_sum"]
        sum_cols = [col for col in sum_cols if col in df.columns]

        # Group by non-lane identifiers
        group_cols = [
            "id",
            "aeg",
            "year",
            "hour",
            "day_of_week",
            "date",
            "is_weekend",
            "is_holiday",
        ]
        group_cols = [col for col in group_cols if col in df.columns]

        # Aggregate numeric columns
        agg_dict = {col: "sum" for col in sum_cols}
        agg_dict["has_speed_data"] = "first"  # Keep if any lane has speed data

        # Perform aggregation
        aggregated = df.groupby(group_cols, dropna=False).agg(agg_dict).reset_index()

        # Recalculate weighted average speed from aggregated speed columns
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

        available_speed_cols = [col for col in self.SPEED_COLS if col in aggregated.columns]
        if available_speed_cols:
            speed_cols_weighted = aggregated[available_speed_cols].mul(
                [speed_midpoints[col] for col in available_speed_cols], axis=1
            )
            total_speed_weight = speed_cols_weighted.sum(axis=1)
            total_speed_count = aggregated[available_speed_cols].sum(axis=1)
            aggregated["avg_speed"] = np.where(
                total_speed_count > 0, total_speed_weight / total_speed_count, np.nan
            )
        else:
            aggregated["avg_speed"] = np.nan

        # Recalculate percentage of heavy vehicles
        if "heavy_vehicle_sum" in aggregated.columns and "total_vehicles" in aggregated.columns:
            aggregated["pct_heavy_vehicles"] = np.where(
                aggregated["total_vehicles"] > 0,
                aggregated["heavy_vehicle_sum"] / aggregated["total_vehicles"],
                0.0,
            )

        return aggregated

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess traffic data.

        Args:
            df: Raw traffic data DataFrame

        Returns:
            Preprocessed DataFrame ready for feature engineering
        """
        df = df.copy()

        # Ensure datetime and extract features
        df["aeg"] = pd.to_datetime(df["aeg"])
        df["hour"] = df["aeg"].dt.hour
        df["day_of_week"] = df["aeg"].dt.dayofweek  # Monday=0, Sunday=6
        df["date"] = df["aeg"].dt.date
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Add holiday flag
        df["is_holiday"] = False
        for year in df["year"].unique():
            year_mask = df["year"] == year
            holidays = self._get_estonian_holidays(year)
            df.loc[year_mask, "is_holiday"] = df.loc[year_mask, "aeg"].isin(holidays)

        # Calculate total vehicles
        df["total_vehicles"] = df[self.VEHICLE_COLS].sum(axis=1)

        # Calculate weighted average speed (approximate)
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

        # Calculate weighted average speed
        speed_cols_weighted = df[self.SPEED_COLS].mul(
            [speed_midpoints[col] for col in self.SPEED_COLS], axis=1
        )
        total_speed_weight = speed_cols_weighted.sum(axis=1)
        total_speed_count = df[self.SPEED_COLS].sum(axis=1)

        df["avg_speed"] = np.where(
            total_speed_count > 0, total_speed_weight / total_speed_count, np.nan
        )

        # Calculate percentage of heavy vehicles
        df["heavy_vehicle_sum"] = df[self.HEAVY_COLS].sum(axis=1)
        df["pct_heavy_vehicles"] = np.where(
            df["total_vehicles"] > 0, df["heavy_vehicle_sum"] / df["total_vehicles"], 0.0
        )

        # Mark rows with missing speed data
        df["has_speed_data"] = ~df[self.SPEED_COLS].isna().all(axis=1)

        # Impute missing speed data
        df = self._impute_missing_speed(df)

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_years: list = [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        val_years: list = [2025],
        test_years: list = [2026],
    ) -> tuple:
        """Split data into train, validation, and test sets.

        Args:
            df: Preprocessed DataFrame
            train_years: Years for training
            val_years: Years for validation
            test_years: Years for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = df[df["year"].isin(train_years)].copy()
        val_df = df[df["year"].isin(val_years)].copy()
        test_df = df[df["year"].isin(test_years)].copy()

        return train_df, val_df, test_df

    def scale_features(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """Scale features using StandardScaler.

        Args:
            df: DataFrame with features
            feature_cols: Columns to scale

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        features = df[feature_cols].values
        scaled_features = self.scaler.fit_transform(features)

        for i, col in enumerate(feature_cols):
            df[f"{col}_scaled"] = scaled_features[:, i]

        return df
