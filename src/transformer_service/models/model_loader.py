"""
Chargement et gestion du modèle Transformer
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available

logger = logging.getLogger(__name__)


class TransformerModelLoader:
    """Classe pour charger et gérer le modèle Transformer"""

    def __init__(
        self,
        model_path: str = "models/transformer",
        encoder_path: str = "data/processed/label_encoder.pkl",
    ):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> None:
        """Charger le modèle Transformer"""
        try:
            logger.info(f"Chargement du modèle Transformer depuis {self.model_path}")
            logger.info(f"Device utilisé: {self.device}")

            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )  # noqa: B615

            # Charger le modèle
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(  # noqa: B615
                    self.model_path
                )
            )
            self.model.to(self.device)
            self.model.eval()  # Mode évaluation

            # Charger l'encodeur de labels
            logger.info(f"Chargement de l'encodeur depuis {self.encoder_path}")
            self.label_encoder = joblib.load(self.encoder_path)  # noqa: B301

            logger.info("✅ Modèle Transformer chargé avec succès")
            logger.info(f" Classes disponibles: {len(self.label_encoder.classes_)}")
            logger.info(f" Paramètres du modèle: {self.model.num_parameters():,}")

        except FileNotFoundError as e:
            logger.error(f"❌ Fichier modèle introuvable: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Prédire la catégorie d'un texte

        Args:
            text: Texte à classifier

        Returns:
            Tuple (catégorie, confiance, scores)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Modèle non chargé")

        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        # Prédiction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Récupérer la prédiction
        predicted_class_id = outputs.logits.argmax().item()
        confidence = probabilities[0][predicted_class_id].item()

        # Décoder la catégorie
        category = self.label_encoder.classes_[predicted_class_id]

        # Créer le dictionnaire de scores
        all_scores = {
            str(self.label_encoder.classes_[i]): float(probabilities[0][i].item())
            for i in range(len(self.label_encoder.classes_))
        }

        # Trier par score décroissant
        all_scores = dict(sorted(all_scores.items(), key=lambda x: x[1], reverse=True))

        return category, confidence, all_scores

    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return (
            self.model is not None
            and self.tokenizer is not None
            and self.label_encoder is not None
        )
