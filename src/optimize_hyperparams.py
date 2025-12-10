"""
Optimisation des hyperparamètres avec Optuna
Semaine 3 - Hyperparameter Tuning
"""
import json
import logging
import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available

from mlflow_configs.mlflow_config import setup_mlflow

sys.path.append(str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Charger les données"""
    train_df = pd.read_csv("data/processed/train.csv")
    label_encoder = joblib.load("data/processed/label_encoder.pkl")  # noqa: B301

    X = train_df["Document_clean"].values
    y = train_df["Topic_encoded"].values

    return X, y, label_encoder


def objective_tfidf(trial: Trial):
    """
    Fonction objectif pour optimiser TF-IDF + SVM
    """

    # Charger les données
    X, y, label_encoder = load_data()

    # Hyper-paramètres à optimiser
    max_features = trial.suggest_int("max_features", 3000, 7000, step=500)
    ngram_1 = trial.suggest_int("ngram_1", 1, 1)
    ngram_2 = trial.suggest_int("ngram_2", 1, 3)
    min_df = trial.suggest_int("min_df", 1, 5)
    max_df = trial.suggest_float("max_df", 0.6, 0.9, step=0.05)

    C = trial.suggest_float("C", 0.1, 10.0, log=True)

    logger.info(f"\n🔬 Trial {trial.number}:")
    logger.info(f"   max_features: {max_features}")
    logger.info(f"   ngram_range: ({ngram_1}, {ngram_2})")
    logger.info(f"   min_df: {min_df}, max_df: {max_df}")
    logger.info(f"   C: {C}")

    # Pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(ngram_1, ngram_2),
                    min_df=min_df,
                    max_df=max_df,
                    stop_words="english",
                ),
            ),
            ("svm", LinearSVC(C=C, random_state=42, max_iter=2000)),
        ]
    )

    # Calibration
    calibrated = CalibratedClassifierCV(pipeline, cv=3)

    try:
        # Cross-validation avec F1-score
        scores = cross_val_score(
            calibrated, X, y, cv=5, scoring="f1_weighted", n_jobs=-1
        )

        mean_score = scores.mean()
        std_score = scores.std()

        logger.info(f"   F1-score: {mean_score:.4f} (+/- {std_score:.4f})")

        return mean_score

    except Exception as e:
        logger.error(f"   Erreur: {e}")
        return 0.0


def optimize_tfidf():
    """Lancer l'optimisation TF-IDF"""

    print("🎯 Optimisation des hyperparamètres TF-IDF + SVM")
    print("=" * 60)

    # MLflow
    setup_mlflow()

    with mlflow.start_run(run_name="optuna_tfidf_tuning") as run:
        logger.info(f"MLflow Run: {run.info.run_id}")

        # Créer l'étude Optuna
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimiser
        study.optimize(objective_tfidf, n_trials=20, n_jobs=-1)

        # Meilleur trial
        best_trial = study.best_trial

        logger.info(f"\n✅ Meilleur trial: {best_trial.number}")
        logger.info(f"   F1-score: {best_trial.value:.4f}")
        logger.info("   Hyperparamètres:")

        best_params = best_trial.params
        for param_name, param_value in best_params.items():
            logger.info(f"      {param_name}: {param_value}")
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("best_f1_score", best_trial.value)

        # Sauvegarder les résultats
        optimization_results = {
            "best_trial": best_trial.number,
            "best_f1_score": best_trial.value,
            "best_params": best_params,
            "num_trials": len(study.trials),
            "timestamp": str(pd.Timestamp.now()),
        }

        with open("metrics/optuna_tfidf_results.json", "w") as f:
            json.dump(optimization_results, f, indent=2)

        logger.info("\n✅ Résultats sauvegardés")

        return best_params


if __name__ == "__main__":
    best_params = optimize_tfidf()

    # Sauvegarder les meilleurs paramètres pour mise à jour de params.yaml
    print("\n📝 À ajouter à params.yaml:")
    print(
        f"""
tfidf_train_optimized:
  max_features: {best_params['max_features']}
  ngram_range: [1, {best_params['ngram_2']}]
  min_df: {best_params['min_df']}
  max_df: {best_params['max_df']}
  C: {best_params['C']}
"""
    )
