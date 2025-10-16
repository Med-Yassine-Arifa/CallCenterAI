from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    text: str = Field(
        ..., 
        min_length=10,
        max_length=5000,
        description="Texte du ticket à classifier"
    )

    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "My laptop screen is broken and needs immediate repair"
            }
        }
class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    predicted_class: str = Field(..., description="Catégorie prédite")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probabilités par classe")
    processing_time_ms: float = Field(..., description="Temps de traitement en ms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "Hardware",
                "confidence": 0.892,
                "probabilities": {
                    "Hardware": 0.892,
                    "Access": 0.045,
                    "Purchase": 0.032
                },
                "processing_time_ms": 12.5
            }
        }
class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    model_loaded: bool
    service: str
    version: str