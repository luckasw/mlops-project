# Real-Time Traffic Anomaly Detection with MLOps

An end-to-end MLOps pipeline for detecting traffic anomalies from Estonian census data, processing hourly measurements of volume, speed, and vehicle types, enabling proactive traffic management through automated model deployment and monitoring.

## Quick Start

### Prerequisites
- Python 3.12+
- Docker and Docker Compose
- DVC access to the configured Backblaze B2/S3 remote
- Minikube and kubectl, if running on Kubernetes


## Project Structure

```
mlops-project/
├── deploy/                  # Deployment manifests and deployment docs
│   └── k8s/                 # Minikube Kubernetes manifests
├── docs/                    # Project notes, reports, and source data docs
├── scripts/                 # One-off utility and training scripts
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
├── .dvc/                  # DVC configuration
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile             # Docker configuration
├── dvc.yaml              # DVC pipeline
├── pyproject.toml        # Python project configuration
├── main.py               # Main entry point
└── README.md             # This file
```

## Features

### Data Pipeline
- **Ingestion**: Load hourly traffic data from CSV files
- **Preprocessing**: Handle missing speed data, calculate derived features
- **Feature Engineering**: Extract temporal and traffic features

### Anomaly Detection
- **Model**: Isolation Forest for multivariate anomaly detection
- **Features**: 8 engineered features including traffic volume, speed, vehicle composition, temporal patterns
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


## Docker Deployment

### Build and Run

```bash
# Build images
docker compose build

# Start all services
docker compose up -d

# Stop services
docker compose down
```

### Services

- **API**: `http://localhost:8000` - FastAPI microservice
- **Dashboard**: `http://localhost:8501` - Streamlit dashboard
- **MLflow**: `http://localhost:5000` - Model tracking
- **Prefect**: `http://localhost:4200` - Workflow orchestration


## Minikube Deployment

The Kubernetes manifests are in `deploy/k8s/minikube.yaml`. The app image does not bake in the local data or model files. Instead, Minikube mounts this project directory and Kubernetes mounts `data/` and `models/` into the pods.

Start Minikube:

```bash
minikube start
```

In a separate terminal, keep the project mount running:

```bash
mkdir -p logs .kube-data/mlflow .kube-data/prefect
minikube mount "$PWD:/mnt/mlops-project"
```

In the main terminal, build the app image inside Minikube and deploy:

```bash
eval "$(minikube docker-env)"
docker build -t mlops-project:local .
kubectl apply -f deploy/k8s/minikube.yaml
kubectl get pods -n mlops-project
```

Open services:

```bash
minikube service api -n mlops-project
minikube service dashboard -n mlops-project
minikube service mlflow -n mlops-project
minikube service prefect -n mlops-project
```

## Data Analysis

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

## Feature Engineering

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

### Estonian Holidays
- Fixed: New Year's Day (Jan 1), Independence Day (Feb 24), May Day (May 1), Christmas (Dec 24-26)
- Variable: Easter Sunday and Monday (calculated per year)

## API Endpoints

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
  "rolling_avg_24h": 145
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
      "rolling_avg_24h": 145
    }
  ]
}
```

### Get Features
```
GET /features
```

## Dashboard

The Streamlit dashboard provides:
- Interactive data exploration
- Traffic volume and speed visualizations
- Anomaly detection results
- Station map with Folium
- Filtering by year, station, and time period


## Data Versioning (DVC)
```bash
# Configure credentials locally
dvc remote modify --local dvstore access_key_id YOUR_BACKBLAZE_KEY_ID
dvc remote modify --local dvstore secret_access_key YOUR_BACKBLAZE_APPLICATION_KEY
```

## Data Sources

- **Estonian Open Data Portal**: [Liiklusloenduse andmed](https://andmed.eesti.ee/datasets/liiklusloenduse-andmed)
- **Transport Administration**: [Liiklusloendusseadmed](https://andmed.eesti.ee/datasets/liiklusloendusseadmed)

## Team

- Albert Wihler
- Tanel Pastarus
