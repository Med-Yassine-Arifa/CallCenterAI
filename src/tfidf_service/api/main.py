from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import time
import logging
from contextlib import asynccontextmanager
from schemas.prediction import PredictionRequest , PredictionResponse , HealthResponse
from models.model_loader import TFIDFModelLoader
 # Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
 )
logger = logging.getLogger(__name__)
 # Métriques Prometheus
REQUEST_COUNT = Counter(
    'tfidf_requests_total',
    'Total de requêtes au service TF-IDF',
    ['endpoint', 'method', 'status']
 )
REQUEST_DURATION = Histogram(
    'tfidf_request_duration_seconds',
    'Durée des requêtes en secondes',
    ['endpoint']
 )
PREDICTION_COUNT = Counter(
    'tfidf_predictions_total',
    'Nombre total de prédictions',
    ['predicted_class']
 )
# Variable globale pour le modèle
model_loader = TFIDFModelLoader()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    logger.info(" Démarrage du service TF-IDF...")
    try:
        model_loader.load_model()
        logger.info("✅ Service TF-IDF prêt!")
    except Exception as e:
        logger.error(f"❌ Erreur au démarrage: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info(" Arrêt du service TF-IDF...")
 # Créer l'application FastAPI
app = FastAPI(
    title="CallCenter TF-IDF Classification Service",
    description="Service de classification de tickets avec TF-IDF + SVM",
    version="1.0.0",
    lifespan=lifespan
 )
 # Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
 )
@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine"""
    return {
        "service": "TF-IDF Classification Service",
        "version": "1.0.0",
        "status": "running"
    }
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check du service
    
    Vérifie que le modèle est chargé et prêt
    """
    REQUEST_COUNT.labels(endpoint='/health', method='GET', status='200').inc()
    
    return HealthResponse(
        status="healthy" if model_loader.is_loaded() else "unhealthy",
        model_loaded=model_loader.is_loaded(),
        service="tfidf_service",
        version="1.0.0"
    )
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Prédire la catégorie d'un ticket
    
    Args:
        request: Requête contenant le texte du ticket
        
    Returns:
        Réponse avec la prédiction et les probabilités
        
    Raises:
        HTTPException: Si le modèle n'est pas chargé ou erreur de prédiction
    """
    start_time = time.time()
    
    try:
        # Vérifier que le modèle est chargé
        if not model_loader.is_loaded():
            REQUEST_COUNT.labels(endpoint='/predict', method='POST', status='503').inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé"
            )
        
        # Prédiction
        with REQUEST_DURATION.labels(endpoint='/predict').time():
            category, confidence, probabilities = model_loader.predict(request.text)
        
        # Calculer le temps de traitement
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Métriques
        REQUEST_COUNT.labels(endpoint='/predict', method='POST', status='200').inc()
        PREDICTION_COUNT.labels(predicted_class=category).inc()
        
        # Réponse
        return PredictionResponse(
            predicted_class=category,
            confidence=confidence,
            probabilities=probabilities,
            processing_time_ms=processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        REQUEST_COUNT.labels(endpoint='/predict', method='POST', status='500').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Endpoint Prometheus pour les métriques
    
    Returns:
        Métriques au format Prometheus
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
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
            detail="Modèle non chargé"
        )
    
    return {
        "model_type": "TF-IDF + LinearSVC (calibrated)",
        "classes": list(model_loader.label_encoder.classes_),
        "num_classes": len(model_loader.label_encoder.classes_),
        "model_path": str(model_loader.model_path)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)