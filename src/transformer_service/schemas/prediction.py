"""
Schémas Pydantic pour le service Transformer
"""

from typing import Dict

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Requête de prédiction"""

    text: str = Field(
        ...,
        min_length=10,
        max_length=512,  # Limite du modèle
        description="Texte du ticket à classifier",
    )

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide")
        return v.strip()


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""

    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: Dict[str, float]
    processing_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Réponse du health check"""

    status: str
    model_loaded: bool
    service: str
    version: str
    device: str  # cpu ou cuda
