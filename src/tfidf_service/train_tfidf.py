import json
import time

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import joblib
except ImportError:
    import pickle as joblib

# Assuming mlflow_configs/mlflow_config.py and setup_mlflow exist and work
from mlflow_configs.mlflow_config import setup_mlflow


def load_params():
    """Charger les paramètres depuis params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["tfidf_train"]


def load_data():
    """Charger les données d'entraînement et de test"""
    print(" Chargement des données...")

    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    label_encoder = joblib.load("data/processed/label_encoder.pkl")  # noqa: B301

    print(f"   Train: {len(train_df)} échantillons")
    print(f"   Test: {len(test_df)} échantillons")
    print(f"   Classes: {len(label_encoder.classes_)}")

    return train_df, test_df, label_encoder


def create_tfidf_pipeline(params):
    """
    Créer le pipeline TF-IDF + SVM.
    Retourne base_pipeline (pour accès TF-IDF) et pipeline calibré.
    """
    print(" Création du pipeline TF-IDF + SVM...")

    base_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=params["max_features"],
                    ngram_range=tuple(params["ngram_range"]),
                    min_df=params["min_df"],
                    max_df=params["max_df"],
                    stop_words="english",
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            ("svm", LinearSVC(random_state=42, max_iter=2000, dual=False)),
        ]
    )

    calibrated_pipeline = CalibratedClassifierCV(base_pipeline, method="sigmoid", cv=3)

    print("✅ Pipeline créé")
    return base_pipeline, calibrated_pipeline


# 🛑 FIX APPLIED HERE: Fit base_pipeline first to extract the fitted vectorizer safely.
def train_and_evaluate(
    base_pipeline, pipeline, train_df, test_df, label_encoder, params
):
    """Entraîner et évaluer le modèle"""
    print(" Entraînement du modèle...")

    X_train, y_train = train_df["Document_clean"], train_df["Topic_encoded"]
    X_test, y_test = test_df["Document_clean"], test_df["Topic_encoded"]

    # 1. Fit the base_pipeline to get the *fitted* TfidfVectorizer for feature counting.
    # We fit it here so the feature count is correct and we can save the fitted vectorizer.
    print("   Pré-entraînement du Vectorizer...")
    base_pipeline.fit(X_train, y_train)

    # Access the TfidfVectorizer from the now-fitted base_pipeline
    tfidf_vectorizer = base_pipeline.named_steps["tfidf"]
    n_features = len(tfidf_vectorizer.get_feature_names_out())

    # 2. Fit the CalibratedClassifierCV pipeline (which will internally fit copies of the base_pipeline)
    print("   Entraînement du CalibratedClassifierCV...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)  # 'pipeline' is the CalibratedClassifierCV wrapper
    training_time = time.time() - start_time
    print(f"   Temps d'entraînement: {training_time:.2f}s")

    print(" Évaluation du modèle...")
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    # The line causing the error is now safely removed/replaced by the steps above.

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "train_f1_macro": f1_score(y_train, train_pred, average="macro"),
        "train_f1_weighted": f1_score(y_train, train_pred, average="weighted"),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_f1_macro": f1_score(y_test, test_pred, average="macro"),
        "test_f1_weighted": f1_score(y_test, test_pred, average="weighted"),
        "test_precision_macro": precision_score(y_test, test_pred, average="macro"),
        "test_recall_macro": recall_score(y_test, test_pred, average="macro"),
        "training_time": training_time,
        "n_features": n_features,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_classes": len(label_encoder.classes_),
    }

    print(f"   Accuracy test: {metrics['test_accuracy']:.4f}")
    print(f"   F1-score macro test: {metrics['test_f1_macro']:.4f}")
    print(f"   Features extraites: {metrics['n_features']}")

    class_report = classification_report(
        y_test, test_pred, target_names=label_encoder.classes_, output_dict=True
    )

    conf_matrix = confusion_matrix(y_test, test_pred)

    # Return the fitted base_pipeline along with the calibrated_pipeline
    return base_pipeline, pipeline, metrics, class_report, conf_matrix


# 🛑 FIX APPLIED HERE: Access fitted vectorizer from base_pipeline
def save_model_and_metrics(
    base_pipeline, pipeline, metrics, class_report, conf_matrix, label_encoder, params
):
    """Sauvegarder modèle, vectorizer et métriques"""
    print(" Sauvegarde du modèle et métriques...")

    # Save the full CalibratedClassifierCV pipeline
    with open("models/tfidf_model.pkl", "wb") as f:
        joblib.dump(pipeline, f)

    # Use the fitted vectorizer from the fitted base_pipeline
    vectorizer = base_pipeline.named_steps["tfidf"]
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        joblib.dump(vectorizer, f)

    full_metrics = {
        "model_type": "TF-IDF + SVM",
        "parameters": params,
        "performance_metrics": metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": label_encoder.classes_.tolist(),
    }

    with open("metrics/tfidf_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(" Modèle sauvegardé dans models/tfidf_model.pkl")
    print(" Vectorizer sauvegardé dans models/tfidf_vectorizer.pkl")
    print(" Métriques sauvegardées dans metrics/tfidf_metrics.json")


# 🛑 FIX APPLIED HERE: Updated variable assignments to handle new return value from train_and_evaluate
def main():
    """Fonction principale avec MLflow tracking"""
    setup_mlflow()

    params = load_params()
    train_df, test_df, label_encoder = load_data()

    base_pipeline, pipeline = create_tfidf_pipeline(params)

    with mlflow.start_run(run_name="tfidf_svm_training") as run:
        print(f"\n MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(params)
        mlflow.log_param("model_type", "TF-IDF + SVM")
        mlflow.log_param("calibration_method", "sigmoid")

        # Capture the newly returned fitted_base_pipeline
        (
            fitted_base_pipeline,
            trained_pipeline,
            metrics,
            class_report,
            conf_matrix,
        ) = train_and_evaluate(
            base_pipeline, pipeline, train_df, test_df, label_encoder, params
        )

        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            trained_pipeline,
            "tfidf_svm_model",
            registered_model_name="callcenter_tfidf_model",
        )

        mlflow.log_artifact("data/processed/label_encoder.pkl", "encoders")

        # Pass the fitted_base_pipeline to save_model_and_metrics
        save_model_and_metrics(
            fitted_base_pipeline,
            trained_pipeline,
            metrics,
            class_report,
            conf_matrix,
            label_encoder,
            params,
        )

        print("\n✅ ENTRAÎNEMENT TERMINÉ!")
        print(" MLflow UI: http://localhost:5000")
        print(f" Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
