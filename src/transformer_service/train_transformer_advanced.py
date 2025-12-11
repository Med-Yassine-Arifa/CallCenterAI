"""
Fine-tuning avancé du modèle DistilBERT avec MLflow tracking complet
Semaine 3 - Optimisations ML
"""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available

from mlflow_configs.mlflow_config import setup_mlflow

sys.path.append(str(Path(__file__).resolve().parents[2]))
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
# Configuration logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_params():
    """Charger les paramètres depuis params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["transformer_train"]


def load_and_prepare_data():
    """
    Charger et préparer les données avec stratifications avancées
    """
    print("📊 Chargement des données...")

    # Charger les données
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Charger l'encodeur
    label_encoder = joblib.load("data/processed/label_encoder.pkl")  # noqa: B301

    # Statistiques d'équilibre du dataset
    train_distribution = train_df["Topic_encoded"].value_counts().sort_index()
    test_distribution = test_df["Topic_encoded"].value_counts().sort_index()

    logger.info("\n📈 Distribution des classes:")
    logger.info(f"   Train: {dict(train_distribution)}")
    logger.info(f"   Test:  {dict(test_distribution)}")

    # Vérifier l'équilibre
    class_weights = train_df["Topic_encoded"].value_counts().sum() / (len(train_df["Topic_encoded"].unique()) * train_df["Topic_encoded"].value_counts())

    logger.info("\n⚖️ Poids des classes (pour équilibrage):")
    for i, weight in class_weights.items():
        logger.info(f"   Classe {i}: {weight:.4f}")

    return train_df, test_df, label_encoder, class_weights.to_dict()


def create_datasets(train_df, test_df, tokenizer, params):
    """
    Créer les datasets avec augmentation optionnelle
    """
    print("🔧 Création des datasets...")

    def tokenize_function(examples):
        """Tokenize avec padding et truncation"""
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=int(params["max_length"]),
            return_tensors=None,
        )

    # Créer datasets Hugging Face
    train_dataset = Dataset.from_dict(
        {
            "text": train_df["Document_clean"].tolist(),
            "labels": train_df["Topic_encoded"].tolist(),
        }
    )

    eval_dataset = Dataset.from_dict(
        {
            "text": test_df["Document_clean"].tolist(),
            "labels": test_df["Topic_encoded"].tolist(),
        }
    )

    # Tokenization
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Split train en train/validation (80/20)
    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)

    logger.info("✅ Datasets créés:")
    logger.info(f"   Train: {len(train_val_split['train'])}")
    logger.info(f"   Validation: {len(train_val_split['test'])}")
    logger.info(f"   Test: {len(eval_dataset)}")

    return train_val_split["train"], train_val_split["test"], eval_dataset


def compute_metrics(eval_pred):
    """Calculer les métriques d'évaluation détaillées"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Métriques principales
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_micro = f1_score(labels, predictions, average="micro")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision": precision,
        "recall": recall,
    }


def train_with_class_weights(model, train_dataset, eval_dataset, label_encoder, params, class_weights):
    """
    Entraîner le modèle avec pondération des classes pour gérer les déséquilibres
    """
    print("🚀 Configuration de l'entraînement avec pondération des classes...")

    # Convertir les poids en tenseurs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(len(label_encoder.classes_))],
        dtype=torch.float,
        device=device,
    )

    logger.info(f"   Device: {device}")
    logger.info(f"   Poids des classes: {weight_tensor}")

    # Arguments d'entraînement
    training_args = TrainingArguments(
        output_dir="models/transformer_model_v2",
        num_train_epochs=int(params["num_epochs"]),
        per_device_train_batch_size=int(params["batch_size"]),
        per_device_eval_batch_size=int(params["batch_size"]) * 2,
        gradient_accumulation_steps=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        eval_steps=100,
        save_strategy="epoch",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
        learning_rate=float(params["learning_rate"]),
        fp16=True,
        tf32=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        seed=42,
        label_smoothing_factor=0.1,
        gradient_checkpointing=True,
    )

    # Créer un trainer personnalisé pour la pondération des classes
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs.get("labels")

            # Appliquer les poids des classes
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss

    # Créer le trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0001)],
    )

    logger.info("🏋️ Démarrage de l'entraînement...")
    start_time = time.time()

    # Entraîner
    trainer.train()

    training_time = time.time() - start_time
    logger.info(f"✅ Entraînement terminé en {training_time/60:.2f} minutes")

    return trainer, training_time


def evaluate_model(trainer, eval_dataset, label_encoder):
    """Évaluation détaillée du modèle"""
    print("📊 Évaluation détaillée du modèle...")

    # Prédictions
    predictions = trainer.predict(eval_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = eval_dataset["labels"]

    # Rapport de classification détaillé
    class_report = classification_report(true_labels, pred_labels, target_names=label_encoder.classes_, output_dict=True)

    # Matrice de confusion
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    # Métriques par classe
    per_class_metrics = {}
    for i, class_name in enumerate(label_encoder.classes_):
        per_class_metrics[class_name] = {
            "precision": class_report[class_name]["precision"],
            "recall": class_report[class_name]["recall"],
            "f1-score": class_report[class_name]["f1-score"],
            "support": int(class_report[class_name]["support"]),
        }

    logger.info("\n🎯 Métriques par classe:")
    for class_name, metrics in per_class_metrics.items():
        logger.info(f"   {class_name}:")
        logger.info(f"     Precision: {metrics['precision']:.4f}")
        logger.info(f"     Recall: {metrics['recall']:.4f}")
        logger.info(f"     F1-score: {metrics['f1-score']:.4f}")
        logger.info(f"     Support: {metrics['support']}")

    return class_report, conf_matrix, per_class_metrics


def save_results(
    trainer,
    model,
    tokenizer,
    label_encoder,
    params,
    training_time,
    class_report,
    conf_matrix,
):
    """Sauvegarder les résultats et artefacts"""
    print("💾 Sauvegarde des résultats...")

    # Sauvegarder le modèle
    output_dir = Path("models/transformer_model_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ✅ CORRECTION : Extraire les métriques correctement
    best_metrics = {}
    if hasattr(trainer.state, "best_metric"):
        best_metrics["best_metric"] = trainer.state.best_metric
    if hasattr(trainer.state, "best_model_checkpoint"):
        best_metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint

    # Ajouter les métriques du log history
    if trainer.state.log_history:
        # Trouver la meilleure évaluation
        eval_logs = [log for log in trainer.state.log_history if "eval_loss" in log]
        if eval_logs:
            best_eval = min(eval_logs, key=lambda x: x.get("eval_loss", float("inf")))
            best_metrics.update(
                {
                    "eval_loss": best_eval.get("eval_loss"),
                    "eval_accuracy": best_eval.get("eval_accuracy"),
                    "eval_f1_weighted": best_eval.get("eval_f1_weighted"),
                    "eval_f1_macro": best_eval.get("eval_f1_macro"),
                    "eval_precision": best_eval.get("eval_precision"),
                    "eval_recall": best_eval.get("eval_recall"),
                }
            )

    # Métriques complètes
    final_metrics = {
        "model_type": "DistilBERT Multilingue - Fine-tuned V2",
        "timestamp": datetime.now().isoformat(),
        "training_time_minutes": training_time / 60,
        "parameters": params,
        "performance": best_metrics,  # ✅ CORRECTION
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "num_parameters": model.num_parameters(),
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
    }

    # Sauvegarder JSON
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "transformer_metrics_v2.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"✅ Modèle sauvegardé dans {output_dir}")
    logger.info(f"✅ Métriques sauvegardées dans {metrics_dir}")
    logger.info("\n📊 Meilleures métriques:")
    for key, value in best_metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")


def main():
    """Fonction principale avec MLflow tracking complet"""

    # Configuration
    setup_mlflow()
    params = load_params()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔧 Device: {device}")
    print(f"🔧 Device: {device}")

    if mlflow.active_run():
        mlflow.end_run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"distilbert_finetuning_v2_{timestamp}"
    # Charger données
    train_df, test_df, label_encoder, class_weights = load_and_prepare_data()

    # MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"\n🔬 MLflow Run ID: {run.info.run_id}")
        logger.info(f"🔬 MLflow Run Name: {run_name}")
        # Logger les paramètres
        mlflow.log_params(
            {
                "model_name": params["model_name"],
                "batch_size": int(params["batch_size"]),
                "learning_rate": float(params["learning_rate"]),
                "num_epochs": int(params["num_epochs"]),
                "device": device,
                "num_classes": len(label_encoder.classes_),
                "training_strategy": "weighted_cross_entropy",
            }
        )

        # Charger tokenizer et modèle
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])  # noqa: B615
        model = AutoModelForSequenceClassification.from_pretrained(  # noqa: B615
            params["model_name"],
            num_labels=len(label_encoder.classes_),
            id2label={i: label for i, label in enumerate(label_encoder.classes_)},
            label2id={label: i for i, label in enumerate(label_encoder.classes_)},
        )

        # Créer datasets
        train_dataset, val_dataset, eval_dataset = create_datasets(train_df, test_df, tokenizer, params)

        # Entraîner
        trainer, training_time = train_with_class_weights(model, train_dataset, val_dataset, label_encoder, params, class_weights)

        # Évaluer
        class_report, conf_matrix, per_class_metrics = evaluate_model(trainer, eval_dataset, label_encoder)

        # Sauvegarder
        save_results(
            trainer,
            model,
            tokenizer,
            label_encoder,
            params,
            training_time,
            class_report,
            conf_matrix,
        )

        # Logger les métriques finales
        final_eval = trainer.evaluate()
        for metric_name, metric_value in final_eval.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)

        # Logger les métriques par classe
        for class_name, metrics in per_class_metrics.items():
            for metric_type, value in metrics.items():
                if isinstance(value, float):
                    mlflow.log_metric(f"{class_name}_{metric_type}", value)

        # Logger le modèle
        try:
            logger.info("📦 Enregistrement du modèle dans MLflow...")
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                task="text-classification",  # ✅ CRITIQUE
                name="distilbert_finetuned_v2",
                registered_model_name="callcenter_distilbert_v2",
                await_registration_for=0,
            )
            logger.info("✅ Modèle enregistré dans MLflow")
        except Exception as e:
            logger.warning(f"⚠️ Erreur MLflow: {e}")
            logger.info("   Le modèle reste sauvegardé localement")

        logger.info("\n✅ FINE-TUNING TRANSFORMER AVANCÉ TERMINÉ!")
        logger.info("🔬 MLflow UI: http://localhost:5000")
        logger.info("📊 Métriques finales:")
        logger.info(f"   Accuracy: {final_eval.get('eval_accuracy', 0):.4f}")
        logger.info(f"   F1 Weighted: {final_eval.get('eval_f1_weighted', 0):.4f}")
        logger.info(f"   F1 Macro: {final_eval.get('eval_f1_macro', 0):.4f}")


if __name__ == "__main__":
    main()
