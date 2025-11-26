# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt .

RUN pip install --user --no-warn-script-location -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

LABEL service="ai-agent"
LABEL version="1.0.0"

ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    SERVICE_PORT=8003

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /root/.local /root/.local

WORKDIR /app

COPY --chown=appuser:appuser src/agent_service ./src/agent_service

USER appuser

EXPOSE ${SERVICE_PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit

CMD ["uvicorn", "src.agent_service.api.main:app", "--host", "0.0.0.0", "--port", "8003"]
