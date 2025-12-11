# =========================
# Dockerfile TF-IDF Service
# Multi-stage build
# =========================

# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.11-slim AS builder

LABEL maintainer="callcenter-mlops@example.com"
LABEL stage="builder"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip install  --no-warn-script-location -r requirements.txt

# -------------------------
# Stage 2: Runtime
# -------------------------
FROM python:3.11-slim AS runtime

LABEL service="tfidf-classification"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    SERVICE_PORT=8001

# Create non-root user and app directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/data && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local /usr/local

WORKDIR /app

# Copy source code
COPY  src/tfidf_service ./src/tfidf_service
COPY  src/monitoring ./src/monitoring
# Models and encoders are loaded at runtime if not present
# COPY  models/tfidf_model_optimized.pkl ./models/
# COPY  models/tfidf_vectorizer_optimized.pkl ./models/
# COPY  data/processed/label_encoder.pkl ./data/processed/

USER appuser

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit

CMD ["uvicorn", "src.tfidf_service.api.main_enhanced:app", "--host", "0.0.0.0", "--port", "8001"]

