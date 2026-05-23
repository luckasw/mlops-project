# Dockerfile for Traffic Anomaly Detection MLOps Pipeline

FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir folium streamlit-folium pandas pyarrow scikit-learn joblib numpy fire mlflow prefect plotly streamlit openpyxl

# Install additional packages
RUN pip install --no-cache-dir gunicorn uvicorn fastapi pydantic httpx evidently dvc python-dotenv workalendar holidays

# Create directories
RUN mkdir -p /app/models \
    /app/data/raw \
    /app/data/processed \
    /app/data/parquet \
    /app/reports/drift \
    /app/logs

# Copy application code
COPY . .

# Set default command
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
