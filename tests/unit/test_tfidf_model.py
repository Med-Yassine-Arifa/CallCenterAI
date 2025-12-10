import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available

sys.path.append("src")


class TestTFIDFModel:
    """Tests pour le modèle TF-IDF"""

    @pytest.fixture
    def sample_data(self):
        """Données d'exemple pour les tests"""
        return pd.DataFrame(
            {
                "Document_clean": [
                    "computer not working properly",
                    "password reset required urgently",
                    "printer paper jam error message",
                    "new laptop needed for employee",
                ],
                "Topic_encoded": [0, 1, 0, 2],
            }
        )

    def test_model_file_exists(self):
        """Vérifier que le fichier modèle existe"""
        model_path = Path("models/tfidf_model.pkl")
        assert model_path.exists(), "Fichier modèle TF-IDF manquant"

    def test_model_loading(self):
        """Tester le chargement du modèle"""
        model = joblib.load("models/tfidf_model.pkl")

        assert model is not None
        assert isinstance(model, BaseEstimator)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_label_encoder_loading(self):
        """Tester le chargement de l'encodeur"""
        encoder_path = Path("data/processed/label_encoder.pkl")
        assert encoder_path.exists(), "Encodeur de labels manquant"

        encoder = joblib.load(encoder_path)

        assert encoder is not None
        assert hasattr(encoder, "classes_")
        assert len(encoder.classes_) > 0

    def test_model_prediction(self, sample_data):
        """Tester les prédictions du modèle"""
        # Charger le modèle
        model = joblib.load("models/tfidf_model.pkl")

        # Test de prédiction
        test_texts = sample_data["Document_clean"].tolist()
        predictions = model.predict(test_texts)
        probabilities = model.predict_proba(test_texts)

        # Vérifications
        assert len(predictions) == len(test_texts)
        assert len(probabilities) == len(test_texts)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert probabilities.shape[1] > 0  # Au moins une classe
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Somme = 1

    def test_metrics_file_exists(self):
        """Vérifier que le fichier de métriques existe"""
        metrics_path = Path("metrics/tfidf_metrics.json")
        assert metrics_path.exists(), "Fichier métriques TF-IDF manquant"


class TestDataPreparation:
    """Tests pour la préparation des données"""

    def test_processed_data_exists(self):
        """Vérifier que les données préparées existent"""
        train_path = Path("data/processed/train.csv")
        test_path = Path("data/processed/test.csv")

        assert train_path.exists(), "Fichier train.csv manquant"
        assert test_path.exists(), "Fichier test.csv manquant"

    def test_processed_data_format(self):
        """Vérifier le format des données préparées"""
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")

        # Vérifier les colonnes nécessaires
        required_columns = ["Document_clean", "Topic_encoded"]
        for col in required_columns:
            assert col in train_df.columns, f"Colonne {col} manquante dans train"
            assert col in test_df.columns, f"Colonne {col} manquante dans test"

        # Vérifier que les données ne sont pas vides
        assert len(train_df) > 0, "Dataset train vide"
        assert len(test_df) > 0, "Dataset test vide"

        # Vérifier les types de données
        assert train_df["Topic_encoded"].dtype in ["int64", "int32"]
        assert test_df["Topic_encoded"].dtype in ["int64", "int32"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
