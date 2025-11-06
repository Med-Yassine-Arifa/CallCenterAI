"""
API FastAPI pour le service Transformer
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from ..models.model_loader import TransformerModelLoader
from ..schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Métriques Prometheus
REQUEST_COUNT = Counter(
    "transformer_requests_total",
    "Total requêtes Transformer",
    ["endpoint", "method", "status"],
)

REQUEST_DURATION = Histogram(
    "transformer_request_duration_seconds",
    "Durée requêtes Transformer",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "transformer_predictions_total",
    "Nombre prédictions Transformer",
    ["predicted_class"],
)

# Modèle global
model_loader = TransformerModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion cycle de vie"""
    logger.info("🚀 Démarrage service Transformer...")
    try:
        model_loader.load_model()
        logger.info("✅ Service Transformer prêt!")
    except Exception as e:
        logger.error(f"❌ Erreur démarrage: {e}")
        raise
    yield
    logger.info("🛑 Arrêt service Transformer...")


app = FastAPI(
    title="CallCenter Transformer Classification Service",
    description="Service classification avec DistilBERT multilingue",
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
        "service": "Transformer Classification Service",
        "model": "distilbert-base-multilingual-cased",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check"""
    REQUEST_COUNT.labels(endpoint="/health", method="GET", status="200").inc()

    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        model_loaded=model_loader.is_loaded(),
        service="transformer_service",
        version="1.0.0",
        device=model_loader.device,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Prédire catégorie avec Transformer"""
    start_time = time.time()

    try:
        if not model_loader.is_loaded():
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="503").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé",
            )

        # Prédiction
        with REQUEST_DURATION.labels(endpoint="/predict").time():
            category, confidence, all_scores = model_loader.predict(request.text)

        processing_time_ms = (time.time() - start_time) * 1000

        # Métriques
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
        PREDICTION_COUNT.labels(predicted_class=category).inc()

        return PredictionResponse(
            predicted_class=category,
            confidence=confidence,
            all_scores=all_scores,
            processing_time_ms=processing_time_ms,
            model_name="distilbert-base-multilingual-cased",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Métriques Prometheus"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Informations modèle"""
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "model_type": "DistilBERT Multilingue",
        "model_name": "distilbert-base-multilingual-cased",
        "device": model_loader.device,
        "classes": list(model_loader.label_encoder.classes_),
        "num_classes": len(model_loader.label_encoder.classes_),
        "num_parameters": model_loader.model.num_parameters(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
