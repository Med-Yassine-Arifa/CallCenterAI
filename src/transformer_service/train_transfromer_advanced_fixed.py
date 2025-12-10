"""
Fine-tuning avancé du modèle DistilBERT avec MLflow tracking complet
Corrected and ready-to-run version
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available

from mlflow_configs.mlflow_config import setup_mlflow

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
# Project root helper so imports work when script is run directly
# your provided setup function

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CUDA / performance settings (TF32)
if torch.cuda.is_available():
    # optional TF32 speedups for Ampere+ GPUs (RTX 30/40)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def load_params():
    """Charger les paramètres depuis params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["transformer_train"]


def load_and_prepare_data():
    """Charger et préparer les données"""
    logger.info("📊 Chargement des données...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    label_encoder = joblib.load("data/processed/label_encoder.pkl")  # noqa: B301

    train_distribution = train_df["Topic_encoded"].value_counts().sort_index()
    test_distribution = test_df["Topic_encoded"].value_counts().sort_index()

    logger.info("📈 Distribution des classes:")
    logger.info(f"   Train: {dict(train_distribution)}")
    logger.info(f"   Test : {dict(test_distribution)}")

    # compute class weights (inverse-frequency style)
    class_weights = train_df["Topic_encoded"].value_counts().sum() / (
        len(train_df["Topic_encoded"].unique())
        * train_df["Topic_encoded"].value_counts()
    )

    logger.info("⚖️ Poids des classes (pour équilibrage):")
    for i, w in class_weights.items():
        logger.info(f"   Classe {i}: {w:.4f}")

    return train_df, test_df, label_encoder, class_weights.to_dict()


def create_datasets(train_df, test_df, tokenizer, params):
    """Créer datasets HF tokenized and split train->train/val"""
    max_length = int(params.get("max_length", 256))
    logger.info(f"🔧 Tokenizer max_length = {max_length}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )

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

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )
    eval_dataset = eval_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    train_val_split = train_dataset.train_test_split(test_size=0.2, seed=42)

    logger.info("✅ Datasets créés:")
    logger.info(f"   Train: {len(train_val_split['train'])}")
    logger.info(f"   Validation: {len(train_val_split['test'])}")
    logger.info(f"   Test: {len(eval_dataset)}")

    return train_val_split["train"], train_val_split["test"], eval_dataset


def compute_metrics(eval_pred):
    """Compute metrics and return keys prefixed with 'eval_' for HF Trainer compatibility"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)

    return {
        "eval_accuracy": accuracy,
        "eval_f1_weighted": f1_weighted,
        "eval_f1_macro": f1_macro,
        "eval_f1_micro": f1_micro,
        "eval_precision": precision,
        "eval_recall": recall,
    }


def train_with_class_weights(
    model, train_dataset, eval_dataset, label_encoder, params, class_weights
):
    """Train with class weighting (a WeightedTrainer that ensures labels and weights share device)"""
    logger.info("🚀 Configuration de l'entraînement avec pondération des classes...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    weight_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(len(label_encoder.classes_))],
        dtype=torch.float,
    ).to(device)

    logger.info(f"Poids des classes (tensor on device): {weight_tensor}")

    # TrainingArguments (fixed keys)
    training_args = TrainingArguments(
        output_dir="models/transformer_model_v3",
        num_train_epochs=int(params.get("num_epochs", 5)),
        per_device_train_batch_size=int(params.get("batch_size", 32)),
        per_device_eval_batch_size=int(params.get("batch_size", 32)) * 2,
        gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 2)),
        warmup_steps=int(params.get("warmup_steps", 200)),
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=int(params.get("logging_steps", 50)),
        eval_strategy="epoch",  # correct key
        save_strategy="epoch",  # correct key
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
        learning_rate=float(params.get("learning_rate", 2e-5)),
        fp16=torch.cuda.is_available(),  # enable fp16 when CUDA available
        dataloader_num_workers=int(params.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(params.get("dataloader_pin_memory", True)),
        optim=params.get("optim", "adamw_torch_fused"),
        remove_unused_columns=False,
        seed=int(params.get("seed", 42)),
    )

    # Custom WeightedTrainer ensuring labels moved to same device as weight_tensor
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # Move labels to same device as model/weight tensor
            labels = inputs.get("labels")
            if labels is None:
                outputs = model(**inputs)
                return (outputs.loss, outputs) if return_outputs else outputs.loss

            labels = labels.to(weight_tensor.device)
            # ensure input tensors are on correct device
            for k, v in inputs.items():
                if torch.is_tensor(v) and v.device != weight_tensor.device:
                    inputs[k] = v.to(weight_tensor.device)

            outputs = model(**inputs)
            logits = outputs.logits

            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(params.get("early_stopping_patience", 5)),
                early_stopping_threshold=float(
                    params.get("early_stopping_threshold", 1e-4)
                ),
            )
        ],
    )

    logger.info("🏋️ Démarrage de l'entraînement...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"✅ Entraînement terminé en {training_time/60:.2f} minutes")

    return trainer, training_time


def evaluate_model(trainer, eval_dataset, label_encoder):
    """Detailed evaluation, returns classification report and confusion matrix"""
    logger.info("📊 Évaluation détaillée du modèle...")
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = eval_dataset["labels"]

    class_report = classification_report(
        true_labels,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(true_labels, preds)

    per_class_metrics = {}
    for i, class_name in enumerate(label_encoder.classes_):
        rr = class_report.get(
            class_name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
        )
        per_class_metrics[class_name] = {
            "precision": rr["precision"],
            "recall": rr["recall"],
            "f1-score": rr["f1-score"],
            "support": int(rr["support"]),
        }

    logger.info("🎯 Métriques par classe calculées")
    return class_report, conf_matrix, per_class_metrics


def safe_log_params(params_dict):
    """Log params once (avoid MLflow error if key already present)"""
    active = mlflow.active_run()
    existing = {}
    if active is not None:
        existing = dict(active.data.params)

    for k, v in params_dict.items():
        if str(k) not in existing:
            try:
                mlflow.log_param(k, v)
            except Exception:
                # ignore any param logging error to avoid failing the run
                logger.warning(f"Could not log param {k}")


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
    """Save model, tokenizer and metrics JSON"""
    logger.info("💾 Sauvegarde des résultats...")

    output_dir = Path("models/transformer_model_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Retrieve best metric safely
    best_metrics = {}
    if hasattr(trainer.state, "best_metric"):
        best_metrics["best_metric"] = trainer.state.best_metric
    if hasattr(trainer.state, "best_model_checkpoint"):
        best_metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint

    # Inspect log_history for useful eval logs (fallback)
    if trainer.state.log_history:
        eval_logs = [
            log
            for log in trainer.state.log_history
            if any(k.startswith("eval_") for k in log.keys())
        ]
        if eval_logs:
            # pick the log with best eval_f1_weighted if present
            best_eval = max(eval_logs, key=lambda x: x.get("eval_f1_weighted", -1.0))
            for k, v in best_eval.items():
                if k.startswith("eval_"):
                    best_metrics[k] = v

    final_metrics = {
        "model_type": "DistilBERT Multilingue - Fine-tuned V2",
        "timestamp": datetime.now().isoformat(),
        "training_time_minutes": training_time / 60,
        "parameters": params,
        "performance": best_metrics,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "num_parameters": model.num_parameters(),
        "num_classes": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "transformer_metrics_v2.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info(f"✅ Modèle sauvegardé dans {output_dir}")
    logger.info(f"✅ Métriques sauvegardées dans {metrics_dir}")


def main():
    # Setup and params
    setup_mlflow()
    params = load_params()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🔧 Device: {device}")
    print(f"🔧 Device: {device}")

    # prepare data
    train_df, test_df, label_encoder, class_weights = load_and_prepare_data()

    # Start run with unique name to avoid re-logging params to same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"distilbert_finetuning_v2_{timestamp}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"🔬 MLflow Run ID: {run.info.run_id}")
        logger.info(f"🔬 MLflow Run Name: {run_name}")

        # safe params logging (avoid crash if logged earlier)
        params_to_log = {
            **params,
            "device": device,
            "num_classes": len(label_encoder.classes_),
            "training_strategy": "weighted_cross_entropy",
        }
        safe_log_params(params_to_log)

        # Load tokenizer/model
        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])  # noqa: B615
        model = AutoModelForSequenceClassification.from_pretrained(  # noqa: B615
            params["model_name"],
            num_labels=len(label_encoder.classes_),
            id2label={i: label for i, label in enumerate(label_encoder.classes_)},
            label2id={label: i for i, label in enumerate(label_encoder.classes_)},
        )

        train_dataset, val_dataset, eval_dataset = create_datasets(
            train_df, test_df, tokenizer, params
        )

        trainer, training_time = train_with_class_weights(
            model, train_dataset, val_dataset, label_encoder, params, class_weights
        )

        class_report, conf_matrix, per_class_metrics = evaluate_model(
            trainer, eval_dataset, label_encoder
        )

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

        final_eval = trainer.evaluate()
        # Log final eval numeric metrics
        for metric_name, metric_value in final_eval.items():
            if isinstance(metric_value, (int, float)):
                try:
                    mlflow.log_metric(metric_name, float(metric_value))
                except Exception:
                    pass

        # Log per-class metrics (flattened)
        for class_name, metrics in per_class_metrics.items():
            for metric_type, value in metrics.items():
                if isinstance(value, (int, float)):
                    try:
                        mlflow.log_metric(f"{class_name}_{metric_type}", float(value))
                    except Exception:
                        pass

        # log the model artifact to MLflow (wrapped)
        try:
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="model",
                task="text-classification",
                name="distilbert_finetuned_v2",
                registered_model_name="callcenter_distilbert_v2",
                await_registration_for=0,
            )
            logger.info("✅ Model successfully logged to MLflow")
        except Exception as exc:
            logger.warning(f"⚠️ MLflow model logging failed: {exc}")
            logger.info(
                "   Model artifacts are saved locally in models/transformer_model_v2"
            )

        # Final logs
        logger.info("✅ FINE-TUNING TRANSFORMER AVANCÉ TERMINÉ!")
        logger.info("🔬 MLflow UI: http://localhost:5000")
        logger.info(
            f"📊 Final eval summary: accuracy={final_eval.get('eval_accuracy', 0):.4f}, "
            f"f1_weighted={final_eval.get('eval_f1_weighted', 0):.4f}"
        )


if __name__ == "__main__":
    main()
