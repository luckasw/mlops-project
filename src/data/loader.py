"""Data loading module for traffic census data."""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd


class TrafficDataLoader:
    """Loads traffic census data from CSV files."""

    DATA_DIR = Path("data/raw")
    YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

    # Vehicle type columns (1-10)
    VEHICLE_COLS = [str(i) for i in range(1, 11)]

    # Speed range columns
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

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """Initialize the data loader.

        Args:
            data_dir: Custom data directory. Defaults to DATA_DIR.
        """
        self.data_dir = Path(data_dir) if data_dir else self.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_year(self, year: int) -> pd.DataFrame:
        """Load data for a specific year.

        Args:
            year: Year to load (2018-2026)

        Returns:
            DataFrame with traffic data for the specified year
        """
        filepath = self.data_dir / f"ll_{year}.csv"

        if not filepath.exists():
            # Try to load from the default data directory
            default_path = Path("data") / f"ll_{year}.csv"
            if default_path.exists():
                filepath = default_path
            else:
                raise FileNotFoundError(f"Data file not found: {filepath}")

        # Define dtype for all columns
        all_numeric_cols = self.VEHICLE_COLS + self.SPEED_COLS + ["kanal"]
        dtype_dict = {col: "Int64" for col in all_numeric_cols}

        df = pd.read_csv(filepath, dtype=dtype_dict, low_memory=False)
        df["aeg"] = pd.to_datetime(df["aeg"])
        df["year"] = year
        return df

    def load_years(self, years: list[int]) -> pd.DataFrame:
        """Load data for multiple years.

        Args:
            years: List of years to load

        Returns:
            Combined DataFrame with all specified years
        """
        dfs = [self.load_year(year) for year in years]
        return pd.concat(dfs, ignore_index=True)

    def load_all(self) -> pd.DataFrame:
        """Load all available years.

        Returns:
            DataFrame with all available traffic data
        """
        return self.load_years(self.YEARS)

    def load_recent(self, n_years: int = 7) -> pd.DataFrame:
        """Load most recent n years of data.

        Args:
            n_years: Number of most recent years to load

        Returns:
            DataFrame with recent data
        """
        years = sorted(self.YEARS)[-n_years:]
        return self.load_years(years)

    def to_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """Save DataFrame to parquet format.

        Args:
            df: DataFrame to save
            path: Output path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, engine="pyarrow")

    def load_parquet(self, path: Path) -> pd.DataFrame:
        """Load DataFrame from parquet format.

        Args:
            path: Path to parquet file

        Returns:
            Loaded DataFrame
        """
        return pd.read_parquet(path, engine="pyarrow")
