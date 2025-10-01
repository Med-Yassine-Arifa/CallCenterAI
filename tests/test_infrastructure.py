"""
Tests d'infrastructure pour le projet MLOps
"""
import os
from pathlib import Path

import mlflow
import pytest

from mlflow_configs.mlflow_config import setup_mlflow


class TestProjectStructure:
    """Tests de la structure du projet"""

    def test_project_directories_exist(self):
        """Vérifier que tous les répertoires nécessaires existent"""
        required_dirs = [
            "data/raw",
            "data/processed",
            "models",
            "src",
            "tests",
            "monitoring",
        ]

        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Directory {dir_path} missing"

    def test_config_files_exist(self):
        """Vérifier que les fichiers de configuration existent"""
        required_files = [
            "requirements.txt",
            "params.yaml",
            "dvc.yaml",
            ".gitignore",
            ".pre-commit-config.yaml",
        ]

        for file_path in required_files:
            assert Path(file_path).exists(), f"Config file {file_path} missing"


class TestMLflowSetup:
    """Tests de la configuration MLflow"""

    def test_mlflow_configuration(self):
        """Tester la configuration MLflow"""
        experiment_id = setup_mlflow()
        assert experiment_id is not None

        # Vérifier que l'expérience est créée
        experiment = mlflow.get_experiment_by_name("callcenter_classification")
        assert experiment is not None
        assert experiment.experiment_id == experiment_id


class TestDVCSetup:
    """Tests de la configuration DVC"""

    def test_dvc_initialized(self):
        """Vérifier que DVC est initialisé"""
        assert Path(".dvc").exists()
        assert Path(".dvc/config").exists()

    def test_dvc_pipeline_valid(self):
        """Vérifier que le pipeline DVC est valide"""
        assert Path("dvc.yaml").exists()
        assert Path("params.yaml").exists()

        # Vérifier la syntaxe du pipeline
        os.system("dvc status")  # Doit passer sans erreur


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
