"""
API de l'Agent IA orchestrateur
"""
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ...monitoring import metrics_collector
from ..schemas.request import AgentRequest, AgentResponse, HealthResponse
from ..services.router import ModelRouter
from ..utils.pii_scrubber import PIIScrubber

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# URLs des services (venant de docker-compose)
TFIDF_BASE_URL = os.getenv("TFIDF_URL", "http://tfidf-service:8001")
TRANSFORMER_BASE_URL = os.getenv("TRANSFORMER_URL", "http://transformer-service:8002")

# Métriques
REQUEST_COUNT = Counter("agent_requests_total", "Total requêtes Agent", ["endpoint", "model_used", "status"])

REQUEST_DURATION = Histogram("agent_request_duration_seconds", "Durée requêtes Agent", ["model_used"])

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

# CORS
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
            transformer_status = "healthy" if transformer_health.status_code == 200 else "unhealthy"
        except Exception:
            transformer_status = "unreachable"

    overall_status = "healthy" if tfidf_status == "healthy" or transformer_status == "healthy" else "unhealthy"

    return HealthResponse(
        status=overall_status,
        service="agent_service",
        version="1.0.0",
        tfidf_service_status=tfidf_status,
        transformer_service_status=transformer_status,
    )


@app.post("/predict", response_model=AgentResponse, tags=["Prediction"])
async def predict(request: AgentRequest):
    """Prédiction intelligente avec routage automatique"""
    start_time = time.time()
    model_used = "unknown"

    try:
        # 1. Scrubbing PII
        scrubbed_text, pii_found = pii_scrubber.scrub(request.text)
        pii_detected = len(pii_found) > 0

        try:
            pii_types = []
            for pii_string in pii_found:
                logger.debug(f"Processing PII string: {pii_string}")
                # Extract type between brackets
                match = re.match(r"\[(\w+)\]:", pii_string)
                if match:
                    pii_type = match.group(1).lower()
                    pii_types.append(pii_type)
                    logger.debug(f"Extracted PII type: {pii_type}")
                else:
                    logger.warning(f"Could not extract PII type from: {pii_string}")

            logger.info(f"Final PII types to record: {pii_types}")

            if pii_types:
                metrics_collector.metrics_collector.record_pii(pii_types)
                logger.warning(f"PII détectées et enregistrées: {pii_types}")
            else:
                logger.warning("No PII types extracted despite PII being detected")

        except Exception as e:
            logger.error(f"Erreur PII metrics: {e}", exc_info=True)

        # 2. Routage intelligent
        model, url, routing_explanation = model_router.choose_model(scrubbed_text)
        model_used = model if model else "unknown"

        # Record routing metrics
        try:
            if isinstance(routing_explanation, dict):
                language = routing_explanation.get("language", "unknown")
                reason = f"{language}_routing"
            elif isinstance(routing_explanation, str):
                reason = routing_explanation
            else:
                reason = "default"

            metrics_collector.metrics_collector.record_routing(model=model_used, reason=reason)
            logger.info(f"Routing: {model_used} - {reason}")

        except Exception as e:
            logger.error(f"Erreur routing metrics: {e}")

        # 3. Appel au service de prédiction
        async with httpx.AsyncClient(timeout=30.0) as client:
            with REQUEST_DURATION.labels(model_used=model_used).time():
                response = await client.post(f"{url}/predict", json={"text": scrubbed_text})

            if response.status_code != 200:
                REQUEST_COUNT.labels(
                    endpoint="/predict",
                    model_used=model_used,
                    status=str(response.status_code),
                ).inc()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Erreur du service {model_used}: {response.text}",
                )

            prediction_data = response.json()

        # 4. Construction de la réponse
        processing_time_ms = (time.time() - start_time) * 1000
        REQUEST_COUNT.labels(endpoint="/predict", model_used=model_used, status="200").inc()

        try:
            metrics_collector.metrics_collector.record_latency(model=model_used, latency=processing_time_ms / 1000)
        except Exception as e:
            logger.error(f"Erreur latency metrics: {e}")

        return AgentResponse(
            predicted_class=prediction_data["predicted_class"],
            confidence=prediction_data["confidence"],
            model_used=model_used,
            pii_detected=pii_detected,
            pii_details=pii_found,
            routing_explanation=routing_explanation,
            processing_time_ms=processing_time_ms,
            prediction_details=prediction_data,
        )

    except httpx.HTTPError as e:
        logger.error(f"Erreur HTTP: {e}")
        REQUEST_COUNT.labels(endpoint="/predict", model_used=model_used, status="503").inc()
        try:
            metrics_collector.metrics_collector.record_error("agent", "service_unavailable")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service backend indisponible: {str(e)}",
        )

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        REQUEST_COUNT.labels(endpoint="/predict", model_used=model_used, status="500").inc()
        try:
            metrics_collector.metrics_collector.record_error("agent", "prediction_error")
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8003)
