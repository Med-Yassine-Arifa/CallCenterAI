"""
Chargement et gestion du modèle TF-IDF
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class TFIDFModelLoader:
    """Classe pour charger et gérer le modèle TF-IDF"""

    def __init__(
        self,
        model_path: str = "models/tfidf_model_optimized.pkl",
        encoder_path: str = "data/processed/label_encoder.pkl",
    ):
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.model = None
        self.label_encoder = None

    def load_model(self) -> None:
        """Charger le modèle et l'encodeur depuis les fichiers"""
        try:
            logger.info(f"Chargement du modèle depuis {self.model_path}")
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            logger.info(f"Chargement de l'encodeur depuis {self.encoder_path}")
            with open(self.encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

            logger.info("✅ Modèle TF-IDF chargé avec succès")
            logger.info(f"Classes disponibles : {len(self.label_encoder.classes_)}")

        except FileNotFoundError as e:
            logger.error(f"❌ Fichier modèle introuvable : {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Prédire la catégorie d'un texte

        Args:
            text: Texte à classifier

        Returns:
            Tuple (catégorie, confiance, probabilités)
        """
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")

        # Prédiction
        prediction_id = self.model.predict([text])[0]
        probas = self.model.predict_proba([text])[0]

        # Décoder la catégorie
        category = self.label_encoder.inverse_transform([prediction_id])[0]
        confidence = float(max(probas))

        # Créer le dictionnaire de probabilités
        probabilities = {
            str(cat): float(prob)
            for cat, prob in zip(self.label_encoder.classes_, probas)
        }

        # Trier par probabilité décroissante
        probabilities = dict(
            sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        )

        return category, confidence, probabilities

    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self.model is not None and self.label_encoder is not None
