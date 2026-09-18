# DeliveryGuard AI — API Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost/shap (compilers not needed for prebuilt wheels, but
# keep libgomp for xgboost's OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/synthetic/ ./data/synthetic/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
