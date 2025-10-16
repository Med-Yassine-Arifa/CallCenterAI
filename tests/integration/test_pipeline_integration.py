import json
import subprocess
from pathlib import Path

import pytest


class TestPipelineIntegration:
    """Tests d'intégration du pipeline"""

    def test_data_preparation_to_tfidf(self):
        """Tester l'enchaînement préparation -> TF-IDF"""

        # Vérifier que la préparation a généré les bons fichiers
        train_path = Path("data/processed/train.csv")
        test_path = Path("data/processed/test.csv")
        encoder_path = Path("data/processed/label_encoder.pkl")

        assert train_path.exists()
        assert test_path.exists()
        assert encoder_path.exists()

        # Vérifier que TF-IDF peut utiliser ces fichiers
        model_path = Path("models/tfidf_model.pkl")
        metrics_path = Path("metrics/tfidf_metrics.json")

        if model_path.exists() and metrics_path.exists():
            # Charger et vérifier les métriques
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            assert "performance_metrics" in metrics
            assert "test_accuracy" in metrics["performance_metrics"]
            assert metrics["performance_metrics"]["test_accuracy"] > 0

    def test_mlflow_experiment_tracking(self):
        """Tester que MLflow track bien les expériences"""
        import mlflow

        from mlflow_configs.mlflow_config import setup_mlflow

        # Configurer MLflow
        setup_mlflow()

        # Vérifier que l'expérience existe
        experiment = mlflow.get_experiment_by_name("callcenter_classification")
        assert experiment is not None

        # Vérifier qu'il y a des runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0, "Aucune run MLflow trouvée"

    def test_dvc_pipeline_status(self):
        """Tester le status du pipeline DVC"""

        # Exécuter dvc status
        result = subprocess.run(
            ["dvc", "status"], capture_output=True, text=True, cwd="."
        )

        # Le pipeline doit être valide (pas d'erreur)
        assert result.returncode == 0, f"Erreur DVC: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
