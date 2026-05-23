# Minikube Deployment

This setup runs the project on Minikube with four services:

- API: FastAPI prediction service
- Dashboard: Streamlit dashboard
- MLflow: tracking server
- Prefect: orchestration UI/server

The app image does not include local data or model artifacts. Minikube mounts this
project directory at `/mnt/mlops-project`, and Kubernetes mounts `data/` and
`models/` from there into the API and dashboard pods.

## 1. Start Minikube

```bash
minikube start
```

## 2. Mount The Project Directory

Run this in a separate terminal and keep it running:

```bash
cd /Users/tanelpastarus/Projects/MLOps/mlops-project
mkdir -p logs .kube-data/mlflow .kube-data/prefect
minikube mount "$PWD:/mnt/mlops-project"
```

## 3. Build The App Image Inside Minikube

In your main terminal:

```bash
cd /Users/tanelpastarus/Projects/MLOps/mlops-project
eval "$(minikube docker-env)"
docker build -t mlops-project:local .
```

## 4. Apply Kubernetes Manifests

```bash
kubectl apply -f k8s/minikube.yaml
kubectl get pods -n mlops-project
```

Wait until the pods are `Running` or `Ready`.

## 5. Open Services

Use Minikube service URLs:

```bash
minikube service api -n mlops-project
minikube service dashboard -n mlops-project
minikube service mlflow -n mlops-project
minikube service prefect -n mlops-project
```

Or use fixed NodePorts:

```text
API:       http://$(minikube ip):30800
Dashboard: http://$(minikube ip):30851
MLflow:    http://$(minikube ip):30500
Prefect:   http://$(minikube ip):30420
```

## Useful Commands

```bash
kubectl get all -n mlops-project
kubectl logs -f deployment/api -n mlops-project
kubectl logs -f deployment/dashboard -n mlops-project
kubectl logs -f deployment/mlflow -n mlops-project
kubectl logs -f deployment/prefect -n mlops-project
kubectl describe pod -l app=api -n mlops-project
```

## Rebuild After Code Changes

```bash
eval "$(minikube docker-env)"
docker build -t mlops-project:local .
kubectl rollout restart deployment/api deployment/dashboard -n mlops-project
```

## Stop And Clean Up

```bash
kubectl delete -f k8s/minikube.yaml
minikube stop
```
