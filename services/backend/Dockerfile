# Backend service Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps (if needed by pandas/numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy service code
COPY app.py /app/app.py

# Expose port
EXPOSE 5001

# Environment variables for model paths (can be overridden in compose)
ENV MODEL_PATH=/app/energy_consumption_model.pkl \
    FEATURE_NAMES_PATH=/app/feature_names.pkl \
    DB_SERVICE_URL=http://db_service:5002/predictions

# Default command - run with production WSGI server
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5001", "app:app"]
