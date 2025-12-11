"""
Test de configuration MLflow
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlflow_configs.mlflow_config import setup_mlflow


def test_mlflow_tracking():
    """Test basique de tracking MLflow"""

    # Configurer MLflow
    setup_mlflow()

    # Données de test
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Démarrer une run MLflow
    with mlflow.start_run(run_name="test_setup") as run:
        # Entraîner un modèle simple
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        # Calculer accuracy
        accuracy = model.score(X, y)

        # Logger dans MLflow
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("n_samples", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", accuracy)

        # Sauvegarder le modèle
        mlflow.sklearn.log_model(model, "test_model", registered_model_name="test_callcenter_model")

        print(f"✅ Run ID: {run.info.run_id}")
        print(f"✅ Accuracy: {accuracy:.4f}")
        print("✅ Modèle sauvegardé dans MLflow")


if __name__ == "__main__":
    test_mlflow_tracking()
