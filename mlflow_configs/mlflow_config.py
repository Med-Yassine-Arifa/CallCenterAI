"""
Configuration MLflow pour le projet CallCenter MLOps (Windows)
"""

from pathlib import Path

import mlflow

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()

# Configuration par défaut
DEFAULT_EXPERIMENT_NAME = "callcenter_classification"


def setup_mlflow():
    """Configure MLflow pour le projet"""
    # Configurer l'URI de tracking
    mlflow.set_tracking_uri(str(MLFLOW_TRACKING_URI))

    # Créer ou récupérer l'expérience par défaut
    experiment = mlflow.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
    if experiment is None:
        # Crée l'expérience avec l'emplacement des artifacts correct
        experiment_id = mlflow.create_experiment(
            name=DEFAULT_EXPERIMENT_NAME, artifact_location=str(MLFLOW_ARTIFACT_ROOT)
        )
    else:
        experiment_id = experiment.experiment_id

    # Définir l'expérience active
    mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)

    print(f"MLflow configuré - Expérience: {DEFAULT_EXPERIMENT_NAME}")
    print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Artifact Root: {MLFLOW_ARTIFACT_ROOT}")

    return experiment_id
