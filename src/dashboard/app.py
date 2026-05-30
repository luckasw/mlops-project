"""Streamlit dashboard for traffic anomaly detection."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


class TrafficDashboard:
    """Streamlit dashboard for visualizing traffic anomalies."""

    MAX_DISPLAY_ROWS = 200_000

    def __init__(self):
        """Initialize the dashboard."""
        self.model = None
        self.data = None
        self.anomalies = None

    def load_data(self, years: list[int], stations: list[str] | None = None) -> pd.DataFrame:
        """Load traffic data for specified years."""
        processed_path = Path("data/processed/traffic_data_processed.parquet")
        feature_cols = [
            "id",
            "aeg",
            "year",
            "total_vehicles",
            "avg_speed",
            "pct_heavy_vehicles",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_holiday",
            "rolling_avg_24h",
            "lane_ratio",
        ]

        if processed_path.exists():
            filters = [("year", "in", years)]
            if stations:
                filters.append(("id", "in", stations))

            try:
                df = pd.read_parquet(
                    processed_path,
                    columns=feature_cols,
                    filters=filters,
                    engine="pyarrow",
                )
            except Exception:
                df = pd.read_parquet(processed_path, columns=feature_cols, engine="pyarrow")
                df = df[df["year"].isin(years)]
                if stations:
                    df = df[df["id"].isin(stations)]
        else:
            from src.data.loader import TrafficDataLoader
            from src.data.preprocessor import TrafficDataPreprocessor
            from src.features.engineer import TrafficFeatureEngineer

            loader = TrafficDataLoader()
            df = loader.load_years(years)
            if stations:
                df = df[df["id"].isin(stations)]

            preprocessor = TrafficDataPreprocessor(impute_speed=True)
            df = preprocessor.preprocess(df)

            engineer = TrafficFeatureEngineer()
            df = engineer.engineer_features(df)

        if len(df) > self.MAX_DISPLAY_ROWS:
            df = df.sample(self.MAX_DISPLAY_ROWS, random_state=42).sort_values("aeg")

        return df

    def load_model(self, model_path: Path) -> None:
        """Load trained model."""
        from src.models.anomaly import TrafficAnomalyDetector

        self.model = TrafficAnomalyDetector()
        try:
            self.model.load(model_path)
        except FileNotFoundError:
            st.warning(f"Model not found at {model_path}. Train a model first.")
            self.model = None

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in the data."""
        if self.model is None:
            st.error("Model not loaded. Cannot detect anomalies.")
            return df

        # Use the model's features
        features = self.model.features_

        # Check if all required features are present
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            st.error(
                f"Missing features: {missing_features}. Please retrain the model with current code."
            )
            return df

        results = self.model.detect_anomalies(df[features])
        # Merge anomaly results back with original data
        return pd.concat(
            [
                df,
                results[
                    [
                        "anomaly_score",
                        "anomaly_prediction",
                        "is_anomaly",
                        "anomaly_score_normalized",
                    ]
                ],
            ],
            axis=1,
        )

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(page_title="Traffic Anomaly Detection", page_icon="🚦", layout="wide")

        st.title("🚦 Real-Time Traffic Anomaly Detection")
        st.markdown("---")

        # Sidebar
        with st.sidebar:
            st.header("Settings")

            # Data selection
            st.subheader("Data Selection")
            available_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
            selected_years = st.multiselect(
                "Select years", available_years, default=[2025, 2026], key="years"
            )

            # Model selection
            st.subheader("Model")
            st.info("Model auto-loaded from models/isolation_forest.pkl")

            # Station filter with real names
            st.subheader("Filters")
            all_stations = self._get_all_stations(selected_years) if selected_years else []

            # Load station name mapping for display
            station_names = self._load_station_names()

            # Create display names mapping
            display_names = {sid: f"{station_names.get(sid, sid)} ({sid})" for sid in all_stations}

            selected_stations_display = st.multiselect(
                "Select stations (with real names)",
                [display_names.get(sid, sid) for sid in all_stations],
                default=[display_names.get(sid, sid) for sid in all_stations[:5]]
                if all_stations
                else [],
                key="stations",
            )

            # Extract just the IDs from the display names
            selected_stations = []
            for display in selected_stations_display:
                # Extract ID from "Name (id)" format
                if "(" in display and ")" in display:
                    sid = display.split("(")[1].rstrip(")")
                    selected_stations.append(sid)
                else:
                    selected_stations.append(display)

        # Auto-load model if not loaded
        if self.model is None:
            default_model_path = Path("models/isolation_forest.pkl")
            if default_model_path.exists():
                self.load_model(default_model_path)

        # Main content
        if selected_years:
            with st.spinner("Loading data..."):
                self.data = self.load_data(selected_years, selected_stations)

            if self.data is not None:
                st.subheader(f"Data Overview")
                st.write(f"Total rows: {len(self.data):,}")
                st.write(f"Date range: {self.data['aeg'].min()} to {self.data['aeg'].max()}")

                # Auto-detect anomalies if model is loaded
                if self.model and self.data is not None:
                    with st.spinner("Detecting anomalies..."):
                        self.anomalies = self.detect_anomalies(self.data)

                    if self.anomalies is not None:
                        st.success(f"Detected {self.anomalies['is_anomaly'].sum()} anomalies")

                # Visualizations
                self._show_visualizations()

                # Anomaly table
                if self.anomalies is not None:
                    self._show_anomaly_table()

        # Map visualization
        self._show_map()

    def _load_station_names(self) -> dict[str, str]:
        """Load station ID to real name mapping."""
        from pathlib import Path

        name_map = {}
        # Try clean CSV first, then fallback to id mapping
        for mapping_file in ["data/ll_jaamad_clean.csv", "data/ll_jaamad_id_mapping.csv"]:
            mapping_path = Path(mapping_file)
            if mapping_path.exists():
                stations = pd.read_csv(mapping_path)
                for _, row in stations.iterrows():
                    sid = str(row.get("id", ""))
                    name = str(row.get("name", ""))
                    if sid and name:
                        name_map[sid] = name
                break
        return name_map

    def _get_all_stations(self, years: list[int]) -> list[str]:
        """Get list of all station IDs for selected years."""
        try:
            processed_path = Path("data/processed/traffic_data_processed.parquet")
            if processed_path.exists():
                df = pd.read_parquet(
                    processed_path,
                    columns=["id", "year"],
                    filters=[("year", "in", years)],
                    engine="pyarrow",
                )
            else:
                from src.data.loader import TrafficDataLoader

                loader = TrafficDataLoader()
                df = loader.load_years(years[:1])[["id"]]
            return sorted(df["id"].unique().tolist())
        except Exception:
            return []

    def _show_visualizations(self):
        """Show data visualizations."""
        if self.data is None:
            return

        st.subheader("📊 Visualizations")

        cols = st.columns(2)

        with cols[0]:
            # Traffic volume over time
            st.plotly_chart(self._create_volume_plot(), use_container_width=True)

        with cols[1]:
            # Average speed over time
            st.plotly_chart(self._create_speed_plot(), use_container_width=True)

        cols = st.columns(2)

        with cols[0]:
            # Traffic by hour
            st.plotly_chart(self._create_hour_plot(), use_container_width=True)

        with cols[1]:
            # Traffic by day of week
            st.plotly_chart(self._create_dow_plot(), use_container_width=True)

        # Anomaly visualization
        if self.anomalies is not None:
            st.plotly_chart(self._create_anomaly_plot(), use_container_width=True)

    def _create_volume_plot(self):
        """Create traffic volume plot."""
        import plotly.express as px

        df = self.data.copy()
        df["date"] = df["aeg"].dt.date
        daily_volume = df.groupby("date")["total_vehicles"].sum().reset_index()

        return px.line(daily_volume, x="date", y="total_vehicles", title="Daily Traffic Volume")

    def _create_speed_plot(self):
        """Create average speed plot."""
        import plotly.express as px

        df = self.data.copy()
        df["date"] = df["aeg"].dt.date
        daily_speed = df.groupby("date")["avg_speed"].mean().reset_index()

        return px.line(daily_speed, x="date", y="avg_speed", title="Daily Average Speed")

    def _create_hour_plot(self):
        """Create traffic by hour plot."""
        import plotly.express as px

        hourly_volume = self.data.groupby("hour")["total_vehicles"].mean().reset_index()

        return px.bar(hourly_volume, x="hour", y="total_vehicles", title="Average Traffic by Hour")

    def _create_dow_plot(self):
        """Create traffic by day of week plot."""
        import plotly.express as px

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow_volume = self.data.groupby("day_of_week")["total_vehicles"].mean().reset_index()
        dow_volume["day_name"] = dow_volume["day_of_week"].map(lambda x: dow_names[x])

        return px.bar(
            dow_volume, x="day_name", y="total_vehicles", title="Average Traffic by Day of Week"
        )

    def _create_anomaly_plot(self):
        """Create anomaly visualization."""
        import plotly.express as px

        df = self.anomalies.copy()
        df["date"] = df["aeg"].dt.date

        # Mark anomalies
        df["anomaly_text"] = df["is_anomaly"].map({True: "Anomaly", False: "Normal"})

        fig = px.scatter(
            df,
            x="date",
            y="total_vehicles",
            color="anomaly_text",
            title="Traffic Volume with Anomalies",
            color_discrete_map={"Anomaly": "red", "Normal": "blue"},
        )

        return fig

    def _show_anomaly_table(self):
        """Show table of detected anomalies."""
        st.subheader("🚨 Detected Anomalies")

        anomalies = self.anomalies[self.anomalies["is_anomaly"]].copy()

        if len(anomalies) == 0:
            st.info("No anomalies detected.")
            return

        # Sort by anomaly score
        anomalies = anomalies.sort_values("anomaly_score", ascending=True)

        # Load station names for display
        station_names = self._load_station_names()

        # Select columns to display
        display_cols = [
            "aeg",
            "id",
            "total_vehicles",
            "avg_speed",
            "anomaly_score",
            "anomaly_score_normalized",
        ]

        # Format for display
        display_df = anomalies[display_cols].copy()
        display_df["aeg"] = display_df["aeg"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["anomaly_score"] = display_df["anomaly_score"].round(4)
        display_df["anomaly_score_normalized"] = display_df["anomaly_score_normalized"].round(4)

        # Add station name column
        display_df["station_name"] = display_df["id"].map(
            lambda x: station_names.get(str(x), "Unknown")
        )

        # Reorder columns to show name first
        display_cols_with_name = [
            "aeg",
            "station_name",
            "id",
            "total_vehicles",
            "avg_speed",
            "anomaly_score",
            "anomaly_score_normalized",
        ]
        display_df = display_df[display_cols_with_name]

        st.dataframe(display_df, use_container_width=True)

        # Show statistics
        st.write(f"Total anomalies: {len(anomalies)}")
        st.write(f"Average anomaly score: {anomalies['anomaly_score'].mean():.4f}")

    def _show_map(self):
        """Show map of traffic stations with real-life names."""
        st.subheader("🗺️ Traffic Station Map")

        try:
            import folium
            from folium.plugins import MarkerCluster
            from streamlit_folium import folium_static

            # Load station mapping with real names and metadata
            mapping_path = Path("data/ll_jaamad_clean.csv")
            if not mapping_path.exists():
                mapping_path = Path("data/ll_jaamad_id_mapping.csv")

            if mapping_path.exists():
                stations = pd.read_csv(mapping_path)

                # Create map centered on Estonia
                m = folium.Map(location=[59.0, 25.0], zoom_start=7, tiles="CartoDB positron")

                # Create marker cluster for better visualization
                marker_cluster = MarkerCluster(name="Stations").add_to(m)

                # Add station markers with real names
                for _, row in stations.iterrows():
                    if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
                        # Build popup with station details
                        popup_html = f"""
                        <b>{row.get("name", row.get("id", "Unknown"))}</b><br>
                        ID: {row.get("id", "N/A")}<br>
                        Road: {row.get("road_name", "N/A")}<br>
                        County: {row.get("county", "N/A")}<br>
                        """
                        if "road_km" in stations.columns and pd.notna(row.get("road_km")):
                            popup_html += f"Km: {row['road_km']}<br>"
                        if "type" in stations.columns and pd.notna(row.get("type")):
                            popup_html += f"Type: {row['type']}<br>"
                        if "status" in stations.columns and pd.notna(row.get("status")):
                            popup_html += f"Status: {row['status']}<br>"

                        folium.Marker(
                            location=[row["latitude"], row["longitude"]],
                            popup=folium.Popup(popup_html, max_width=300),
                            tooltip=row.get("name", row.get("id", "Unknown")),
                            icon=folium.Icon(icon="road", prefix="fa"),
                        ).add_to(marker_cluster)

                # Add layer control
                folium.LayerControl().add_to(m)

                folium_static(m, width=800, height=500)
            else:
                st.warning(
                    "Station mapping file not found. Expected at data/ll_jaamad_clean.csv or data/ll_jaamad_id_mapping.csv"
                )
        except ImportError:
            st.warning(
                "Folium or streamlit-folium not available. Install with: uv add folium streamlit-folium"
            )


def run_dashboard():
    """Run the traffic anomaly detection dashboard."""
    dashboard = TrafficDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()
