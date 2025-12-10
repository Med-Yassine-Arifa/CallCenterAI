"""
Service TF-IDF amélioré avec métriques métier
"""
import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from ...monitoring.metrics_collector import metrics_collector
from ..models.model_loader import TFIDFModelLoader
from ..schemas.prediction import HealthResponse, PredictionRequest, PredictionResponse

sys.path.insert(0, str(__file__).replace("src/tfidf_service/api/main_enhanced.py", ""))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_loader = TFIDFModelLoader()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage service TF-IDF...")
    try:
        model_loader.load_model()
        logger.info("✅ Service TF-IDF prêt!")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        raise
    yield
    logger.info("🛑 Arrêt service...")


app = FastAPI(
    title="CallCenter TF-IDF Classification Service",
    description="Service avec métriques métier avancées",
    version="1.1.0",
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
        "service": "TF-IDF Classification Service",
        "version": "1.1.0",
        "metrics": "enabled",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        model_loaded=model_loader.is_loaded(),
        service="tfidf_service",
        version="1.1.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """Prédiction avec métriques métier"""
    start_time = time.time()

    try:
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé",
            )

        # Prédiction
        category, confidence, probabilities = model_loader.predict(request.text)

        processing_time_ms = (time.time() - start_time) * 1000

        # Enregistrer les métriques métier
        metrics_collector.record_prediction(
            service="tfidf",
            class_name=category,
            confidence=confidence,
            processing_time=processing_time_ms / 1000,
        )

        # Mettre à jour les statistiques
        metrics_collector.update_statistics("tfidf")

        return PredictionResponse(
            predicted_class=category,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms,
        )

    except HTTPException:
        metrics_collector.record_error("tfidf", "service_unavailable")
        raise
    except Exception as e:
        logger.error(f"Erreur: {e}")
        metrics_collector.record_error("tfidf", "prediction_error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Informations sur le modèle chargé

    Returns:
        Informations sur les classes et le modèle
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé",
        )

    return {
        "model_type": "TF-IDF + LinearSVC (calibrated)",
        "classes": list(model_loader.label_encoder.classes_),
        "num_classes": len(model_loader.label_encoder.classes_),
        "model_path": str(model_loader.model_path),
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Métriques Prometheus incluant les métriques métier"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
