# Real-Time Traffic Anomaly Detection with MLOps

An end-to-end MLOps pipeline for detecting traffic anomalies from Estonian census data, processing hourly measurements of volume, speed, and vehicle types, enabling proactive traffic management through automated model deployment and monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- UV package manager (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mlops-project.git
cd mlops-project

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## 📁 Project Structure

```
mlops-project/
├── data/                    # Raw and processed data
│   ├── raw/                # Raw CSV files
│   ├── processed/          # Processed parquet files
│   └── ll_jaamad_*.csv     # Station mapping files
├── models/                 # Trained models
├── reports/                # Monitoring reports
│   └── drift/              # Drift detection reports
├── logs/                  # Application logs
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   │   ├── loader.py       # Data loader
│   │   └── preprocessor.py # Data preprocessing
│   ├── features/           # Feature engineering
│   │   └── engineer.py     # Feature engineering
│   ├── models/             # ML models
│   │   └── anomaly.py      # Anomaly detection model
│   ├── api/                # FastAPI microservice
│   │   └── app.py          # API endpoints
│   ├── flows/              # Prefect flows
│   │   └── training.py     # Training and retraining flows
│   ├── monitoring/         # Data drift monitoring
│   │   ├── drift.py        # Drift detection
│   │   └── monitor.py     # Monitoring utilities
│   └── dashboard/          # Streamlit dashboard
│       └── app.py          # Dashboard application
├── tests/                  # Tests
├── configs/                # Configuration files
├── .dvc/                  # DVC configuration
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Docker configuration
├── dvc.yaml              # DVC pipeline
├── pyproject.toml        # Python project configuration
├── main.py               # Main entry point
└── README.md             # This file
```

## 🎯 Features

### Data Pipeline
- **Ingestion**: Load hourly traffic data from CSV files (2018-2026)
- **Preprocessing**: Handle missing speed data, calculate derived features
- **Feature Engineering**: Extract temporal and traffic features

### Anomaly Detection
- **Model**: Isolation Forest for multivariate anomaly detection
- **Features**: 9 engineered features including traffic volume, speed, vehicle composition, temporal patterns
- **Training Strategy**: Incremental learning with 2018-2024 for training, 2025 for validation, 2026 for testing

### MLOps Components
- **MLflow**: Experiment tracking and model registry
- **Prefect**: Workflow orchestration with hourly scheduling
- **Evidently AI**: Data drift detection and monitoring
- **DVC**: Data version control
- **Docker**: Containerized deployment

### Services
- **FastAPI**: REST API for real-time predictions
- **Streamlit + Folium**: Interactive dashboard with map visualization

## 🚗 Usage

### Data Preprocessing

```bash
# Preprocess all data and save to parquet
uv run python main.py preprocess
```

### Model Training

```bash
# Train the anomaly detection model
uv run python main.py train
```

### Run API Service

```bash
# Start FastAPI server
uv run python main.py predict

# Or with uvicorn directly
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Run Dashboard

```bash
# Start Streamlit dashboard
uv run streamlit run src/dashboard/app.py
```

### Monitor Data Drift

```bash
# Check for data drift
uv run python main.py monitor
```

### Retrain Model

```bash
# Retrain with new data
uv run python main.py retrain
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=
MLFLOW_S3_BUCKET=

# Prefect
PREFECT_API_URL=http://localhost:4200/api

# Data
DATA_DIR=./data
MODEL_DIR=./models
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# Stop services
docker-compose down
```

### Services

- **API**: `http://localhost:8000` - FastAPI microservice
- **Dashboard**: `http://localhost:8501` - Streamlit dashboard
- **MLflow**: `http://localhost:5000` - Model tracking
- **Prefect**: `http://localhost:4200` - Workflow orchestration

## 📊 Data Analysis

### Dataset Overview
- **Source**: Estonian Transport Administration (Liiklusloenduse andmed)
- **Period**: 2018-2026 (hourly measurements)
- **Volume**: 14,076,486 rows across 9 yearly CSV files
- **Structure**: Each row = 1 counter device × 1 lane × 1 hour

### Columns
- **Identifiers**: `id`, `kanal`, `aeg` (timestamp)
- **Vehicle Types**: 1-10 (Motorcycle, Car, Heavy Van, Rigid, Articulated HGV, Minibus, Bus/Coach)
- **Speed Ranges**: `<40Kph` to `=>130` (11 speed bins in 10 km/h increments)

### Data Quality
- Missing speed data for stations: `25d51`, `0e481`, `ff2e7`, `fde61`
- Speed data imputed using historical medians per station and hour
- Vehicle type counts sum to speed range counts (validation passed for 2018-2023)

## 🎨 Feature Engineering

### Core Features

| Feature | Description | Type |
|---------|-------------|------|
| `total_vehicles` | Sum of vehicle counts (1-10) | Numerical |
| `avg_speed` | Weighted average speed from speed bins | Numerical |
| `pct_heavy_vehicles` | Percentage of heavy vehicles (6-8) | Numerical |
| `hour` | Hour of day (0-23) | Categorical |
| `day_of_week` | Day of week (0-6, Monday=0) | Categorical |
| `is_weekend` | Weekend flag | Boolean |
| `is_holiday` | Estonian holiday flag | Boolean |
| `rolling_avg_24h` | 24-hour moving average of traffic volume | Numerical |
| `lane_ratio` | Ratio of lane volume to total station volume | Numerical |

### Estonian Holidays
- Fixed: New Year's Day (Jan 1), Independence Day (Feb 24), May Day (May 1), Christmas (Dec 24-26)
- Variable: Easter Sunday and Monday (calculated per year)

## 🔍 API Endpoints

### Health Check
```
GET /health
```

### Single Prediction
```
POST /predict
Content-Type: application/json

{
  "total_vehicles": 150,
  "avg_speed": 85.5,
  "pct_heavy_vehicles": 0.25,
  "hour": 14,
  "day_of_week": 2,
  "is_weekend": false,
  "is_holiday": false,
  "rolling_avg_24h": 145,
  "lane_ratio": 0.5
}
```

### Batch Prediction
```
POST /predict/batch
Content-Type: application/json

{
  "observations": [
    {
      "total_vehicles": 150,
      "avg_speed": 85.5,
      "pct_heavy_vehicles": 0.25,
      "hour": 14,
      "day_of_week": 2,
      "is_weekend": false,
      "is_holiday": false,
      "rolling_avg_24h": 145,
      "lane_ratio": 0.5
    }
  ]
}
```

### Get Features
```
GET /features
```

## 📈 Dashboard

The Streamlit dashboard provides:
- Interactive data exploration
- Traffic volume and speed visualizations
- Anomaly detection results
- Station map with Folium
- Filtering by year, station, and time period

## 🔄 MLOps Pipeline

### Data Versioning (DVC)
```bash
# Initialize DVC
dvc init

# Add data to DVC
dvc add data/ll_2025.csv

# Run pipeline
dvc repro

# Push to remote
dvc push
```

### Experiment Tracking (MLflow)
```python
import mlflow

mlflow.set_experiment("traffic_anomaly_detection")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("anomaly_rate", 0.01)
    mlflow.sklearn.log_model(model, "model")
```

### Workflow Orchestration (Prefect)
```python
from prefect import flow, task

@flow(name="hourly_retraining")
def retrain_flow():
    # Load new data
    # Retrain model
    # Save and deploy
    pass

# Schedule hourly
retrain_flow.schedule(cron="0 * * * *")
```

### Drift Detection (Evidently AI)
```python
from src.monitoring.drift import TrafficDriftDetector

detector = TrafficDriftDetector()
drift_report = detector.detect_drift(reference_data, current_data, features)
```

## 🧪 Testing

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=src
```

## 📚 Data Sources

- **Estonian Open Data Portal**: [Liiklusloenduse andmed](https://andmed.eesti.ee/datasets/liiklusloenduse-andmed)
- **Transport Administration**: [Liiklussagedus](https://transpordiamet.ee/liiklussagedus)

## 👥 Team

- Albert Wihler
- Tanel Pastarus

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request
