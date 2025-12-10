# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install build dependencies (needed for many Python wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip install  --no-warn-script-location \
    torch==2.9.0+cu128 \
    torchvision==0.24.0+cu128 \
    torchaudio==2.9.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --timeout 300


RUN   pip install  --no-warn-script-location -r requirements.txt




# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL service="ai-agent" \
      version="1.0.0"


ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    SERVICE_PORT=8003 \
    AGENT_CACHE=/app/.cache

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Copy installed python packages into runtime image
COPY --from=builder /usr/local /usr/local

WORKDIR /app

COPY src/agent_service ./src/agent_service
COPY src/monitoring ./src/monitoring

USER appuser

EXPOSE ${SERVICE_PORT}



CMD ["uvicorn", "src.agent_service.api.main:app", "--host", "0.0.0.0", "--port", "8003"]