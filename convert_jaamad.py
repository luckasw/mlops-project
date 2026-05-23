#!/usr/bin/env python3
"""Convert LL jaamad.xlsx to clean, machine-readable CSV format."""

from pathlib import Path

import pandas as pd

EXCEL_FILE = Path("data/LL jaamad.xlsx")

# Clean column name mapping
CLEAN_COLUMNS = {
    "ID": "id",
    "Nimetus": "name",
    "Liik": "type",
    "Staatus": "status",
    "Maakond": "county",
    "Tee nimi": "road_name",
    "Tee nr": "road_number",
    "Tee km": "road_km",
    "Lat": "latitude",
    "Lon": "longitude",
    "Kanalid ehk sõidurajad": "lanes_description",
}


def clean_lanes_description(desc):
    """Normalize lane descriptions by replacing Estonian separator with pipe."""
    if pd.isna(desc):
        return ""
    return str(desc).replace(" | ", "|")


def main():
    # Read Excel
    df = pd.read_excel(EXCEL_FILE, engine="openpyxl")

    # Rename columns
    df_clean = df.rename(columns=CLEAN_COLUMNS)

    # Clean lane descriptions
    df_clean["lanes_description"] = df_clean["lanes_description"].apply(clean_lanes_description)

    # Fill missing coordinates with 0.0
    df_clean["latitude"] = df_clean["latitude"].fillna(0.0)
    df_clean["longitude"] = df_clean["longitude"].fillna(0.0)

    # Option 1: Valid IDs only (recommended for mapping)
    df_valid = df_clean.dropna(subset=["id"]).copy()
    df_valid.to_csv(Path("data/ll_jaamad_clean.csv"), index=False, encoding="utf-8")

    # Option 2: All rows with placeholder for missing IDs
    df_all = df_clean.copy()
    df_all["id"] = df_all["id"].fillna("MISSING_ID")
    df_all.to_csv(Path("data/ll_jaamad_all.csv"), index=False, encoding="utf-8")

    # Option 3: Minimal ID mapping
    df_map = df_valid[
        ["id", "name", "latitude", "longitude", "road_name", "road_number", "road_km"]
    ].copy()
    df_map.to_csv(Path("data/ll_jaamad_id_mapping.csv"), index=False, encoding="utf-8")

    print(f"Converted {EXCEL_FILE} to:")
    print(f"  - data/ll_jaamad_clean.csv ({len(df_valid)} stations)")
    print(f"  - data/ll_jaamad_all.csv ({len(df_all)} stations)")
    print(f"  - data/ll_jaamad_id_mapping.csv ({len(df_map)} stations)")


if __name__ == "__main__":
    main()
