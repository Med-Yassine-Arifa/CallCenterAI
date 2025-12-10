# =========================
# Dockerfile Transformer Service
# Multi-stage build
# =========================

# -------------------------
# Stage 1: Builder
# -------------------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
        gcc \
        g++ \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt ./requirements.txt

RUN pip install  --no-warn-script-location \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --timeout 300


RUN   pip install  --no-warn-script-location -r requirements.txt


# -------------------------
# Stage 2: Runtime
# -------------------------
FROM python:3.11-slim AS runtime

LABEL service="transformer-classification"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    SERVICE_PORT=8002 \
    TRANSFORMERS_CACHE=/app/.cache

# Create non-root user and directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/data /app/.cache && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local /usr/local

WORKDIR /app

# Copy source code, models, and data
COPY  src/transformer_service ./src/transformer_service
COPY  src/monitoring ./src/monitoring
COPY  models/transformer_model_v2 ./models/transformer
COPY  data/processed/label_encoder.pkl ./data/processed/

USER appuser

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit

CMD ["uvicorn", "src.transformer_service.api.main:app", "--host", "0.0.0.0", "--port", "8002"]

