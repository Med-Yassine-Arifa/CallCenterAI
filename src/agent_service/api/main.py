"""
API de l'Agent IA orchestrateur
"""
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ..schemas.request import AgentRequest, AgentResponse, HealthResponse
from ..services.router import ModelRouter
from ..utils.pii_scrubber import PIIScrubber

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# URLs des services (venant de docker-compose)
TFIDF_BASE_URL = os.getenv("TFIDF_URL", "http://tfidf-service:8001")
TRANSFORMER_BASE_URL = os.getenv("TRANSFORMER_URL", "http://transformer-service:8002")

# Métriques
REQUEST_COUNT = Counter(
    "agent_requests_total", "Total requêtes Agent", ["endpoint", "model_used", "status"]
)

REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds", "Durée requêtes Agent", ["model_used"]
)

PII_DETECTED_COUNT = Counter("agent_pii_detected_total", "PII détectées")

# Instances globales
pii_scrubber = PIIScrubber()
model_router = ModelRouter(TFIDF_BASE_URL, TRANSFORMER_BASE_URL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage Agent IA...")
    yield
    logger.info("🛑 Arrêt Agent IA...")

app = FastAPI(
    title="CallCenter AI Agent",
    description="Agent IA orchestrateur intelligent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "AI Agent Orchestrator",
        "version": "1.0.0",
        "capabilities": ["PII scrubbing", "Intelligent routing", "Multi-model support"],
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check complet de l'agent et des services"""
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Service TF-IDF
        try:
            tfidf_health = await client.get(f"{TFIDF_BASE_URL}/health")
            tfidf_status = "healthy" if tfidf_health.status_code == 200 else "unhealthy"
        except Exception:
            tfidf_status = "unreachable"

        # Service Transformer
        try:
            transformer_health = await client.get(f"{TRANSFORMER_BASE_URL}/health")
            transformer_status = (
                "healthy" if transformer_health.status_code == 200 else "unhealthy"
            )
        except Exception:
            transformer_status = "unreachable"

    overall_status = (
        "healthy"
        if tfidf_status == "healthy" or transformer_status == "healthy"
        else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        service="agent_service",
        version="1.0.0",
        tfidf_service_status=tfidf_status,
        transformer_service_status=transformer_status,
    )

@app.post("/predict", response_model=AgentResponse, tags=["Prediction"])
async def predict(request: AgentRequest):
    """
    Prédiction intelligente avec routage automatique
    """
    start_time = time.time()

    try:
        # 1. Scrubbing PII
        scrubbed_text, pii_found = pii_scrubber.scrub(request.text)
        pii_detected = len(pii_found) > 0

        if pii_detected:
            PII_DETECTED_COUNT.inc()
            logger.warning(f"PII détectées: {pii_found}")

        # 2. Routage intelligent
        model, url, routing_explanation = model_router.choose_model(scrubbed_text)

        # 3. Appel au service de prédiction
        async with httpx.AsyncClient(timeout=30.0) as client:
            with REQUEST_DURATION.labels(model_used=model).time():
                response = await client.post(
                    f"{url}/predict", json={"text": scrubbed_text}
                )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur du service {model}: {response.text}",
                )

            prediction_data = response.json()

        # 4. Construction de la réponse
        processing_time_ms = (time.time() - start_time) * 1000

        REQUEST_COUNT.labels(endpoint="/predict", model_used=model, status="200").inc()

        return AgentResponse(
            predicted_class=prediction_data["predicted_class"],
            confidence=prediction_data["confidence"],
            model_used=model,
            pii_detected=pii_detected,
            pii_details=pii_found,
            routing_explanation=routing_explanation,
            processing_time_ms=processing_time_ms,
            prediction_details=prediction_data,
        )

    except httpx.HTTPError as e:
        logger.error(f"Erreur HTTP: {e}")
        REQUEST_COUNT.labels(
            endpoint="/predict", model_used="unknown", status="503"
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service backend indisponible: {str(e)}",
        )

    except Exception as e:
        logger.error(f"Erreur: {e}")
        REQUEST_COUNT.labels(
            endpoint="/predict", model_used="unknown", status="500"
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
