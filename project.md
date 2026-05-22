# Real-Time Traffic Anomaly Detection with MLOps

## Abstract
An end-to-end MLOps pipeline that detects traffic anomalies from Estonian census data by processing hourly measurements of volume, speed, and vehicle types, enabling proactive traffic management through automated model deployment and monitoring.

## Workflow
The system continuously ingests new data files, extracts temporal and traffic features, applies an anomaly detection model, exposes predictions through an API, and monitors performance to trigger retraining with fresh data and operator feedback.
   Purpose | Technology |
 |---------|------------|
 | Baseline training | Scikit-learn |
 | Experiment tracking | MLflow |
 | Scheduling | Prefect (hourly job) |
 | Anomaly scoring | FastAPI microservice |
 | Drift detection | Evidently AI |
 | Dashboard | Streamlit + Folium (map) (maybe) |
 | Storage | Parquet |
 | Deployment | Docker |
 | Version control | Git, DVC, MLflow |

 Data: [https://andmed.eesti.ee/datasets/liiklusloenduse-andmed](https://andmed.eesti.ee/datasets/liiklusloenduse-andmed)


Model:
Model should train on all available data and then be able to add data to it. In this project only train up to 2024, then later 2025 can be added by retraining and 2026 stays for testing later. 

Model should

Team: Albert Wihler, Tanel Pastarus
