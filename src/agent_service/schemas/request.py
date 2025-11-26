"""
Schémas pour l'Agent IA
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Requête à l'agent"""

    text: str = Field(..., min_length=10, max_length=5000)
    force_model: Optional[str] = Field(
        None, description="Forcer un modèle (tfidf/transformer)"
    )


class AgentResponse(BaseModel):
    """Réponse de l'agent"""

    predicted_class: str
    confidence: float
    model_used: str
    pii_detected: bool
    pii_details: List[str]
    routing_explanation: Dict
    processing_time_ms: float
    prediction_details: Dict


class HealthResponse(BaseModel):
    """Health check agent"""

    status: str
    service: str
    version: str
    tfidf_service_status: str
    transformer_service_status: str
