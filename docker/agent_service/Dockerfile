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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install all requirements globally (NOT --user)
RUN pip install --no-cache-dir -r requirements.txt


# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL service="ai-agent" \
      version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    SERVICE_PORT=8003

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Copy installed python packages into runtime image
COPY --from=builder /usr/local /usr/local

WORKDIR /app

COPY src/agent_service ./src/agent_service

USER appuser

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

CMD ["uvicorn", "src.agent_service.api.main:app", "--host", "0.0.0.0", "--port", "8003"]
