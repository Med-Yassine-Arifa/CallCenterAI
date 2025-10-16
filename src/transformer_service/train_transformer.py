import json
import pickle
import sys
import time
from pathlib import Path

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from mlflow_configs.mlflow_config import setup_mlflow

sys.path.append(str(Path(__file__).resolve().parents[2]))


def load_params():
    """Charger les paramètres depuis params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["transformer_train"]


def load_data():
    """Charger les données d'entraînement et de test"""
    print("📊 Chargement des données...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    with open("data/processed/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print(f" Train: {len(train_df)} échantillons")
    print(f" Test: {len(test_df)} échantillons")
    print(f" Classes: {len(label_encoder.classes_)}")

    return train_df, test_df, label_encoder


def prepare_datasets(train_df, test_df, tokenizer, params):
    """Préparer les datasets pour Transformer"""
    print("🔧 Préparation des datasets...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=params["max_length"],
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

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    print("✅ Datasets préparés")
    print(f" Train tokenized: {len(train_dataset)}")
    print(f" Eval tokenized: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    """Calculer les métriques d'évaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
    }


def train_transformer_model(train_dataset, eval_dataset, label_encoder, params):
    """Entraîner le modèle Transformer"""
    print("🚀 Configuration du modèle Transformer...")

    model_name = params["model_name"]
    num_labels = len(label_encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )

    print(f" Modèle: {model_name}")
    print(f" Paramètres: {model.num_parameters():,}")
    print(f" Labels: {num_labels}")

    training_args = TrainingArguments(
        output_dir="models/transformer_model",
        num_train_epochs=params["num_epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"] * 2,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_weighted",
        greater_is_better=True,
        learning_rate=float(params["learning_rate"]),
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        push_to_hub=False,
        max_steps=-1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("🏋️ Entraînement du modèle...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f" Temps d'entraînement: {training_time / 60:.2f} minutes")

    return trainer, model, tokenizer


def evaluate_and_save(trainer, model, tokenizer, label_encoder, params):
    """Évaluer le modèle et sauvegarder les résultats"""
    print("📊 Évaluation finale du modèle...")
    eval_results = trainer.evaluate()

    metrics = {
        "final_accuracy": eval_results["eval_accuracy"],
        "final_f1_weighted": eval_results["eval_f1_weighted"],
        "final_f1_macro": eval_results["eval_f1_macro"],
        "final_precision": eval_results["eval_precision"],
        "final_recall": eval_results["eval_recall"],
        "training_time_minutes": trainer.state.log_history[-1].get("train_runtime", 0)
        / 60,
        "num_parameters": model.num_parameters(),
        "num_classes": len(label_encoder.classes_),
    }

    print(f" Accuracy finale: {metrics['final_accuracy']:.4f}")
    print(f" F1-score weighted: {metrics['final_f1_weighted']:.4f}")

    model.save_pretrained("models/transformer_model")
    tokenizer.save_pretrained("models/transformer_model")

    full_metrics = {
        "model_type": "DistilBERT Multilingue",
        "model_name": params["model_name"],
        "parameters": params,
        "performance_metrics": metrics,
        "classes": label_encoder.classes_.tolist(),
    }

    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/transformer_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    print("✅ Modèle sauvegardé dans models/transformer_model/")
    print("✅ Métriques sauvegardées dans metrics/transformer_metrics.json")

    return metrics


def main():
    """Fonction principale avec MLflow tracking"""
    setup_mlflow()
    params = load_params()
    train_df, test_df, label_encoder = load_data()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device utilisé: {device}")

    with mlflow.start_run(run_name="distilbert_multilingual_training") as run:
        print(f"\n🔬 MLflow Run ID: {run.info.run_id}")
        mlflow.log_params(params)
        mlflow.log_param("model_type", "DistilBERT Multilingue")
        mlflow.log_param("device", device)
        mlflow.log_param("num_classes", len(label_encoder.classes_))

        tokenizer = AutoTokenizer.from_pretrained(params["model_name"])
        train_dataset, eval_dataset = prepare_datasets(
            train_df, test_df, tokenizer, params
        )

        trainer, model, tokenizer = train_transformer_model(
            train_dataset, eval_dataset, label_encoder, params
        )

        metrics = evaluate_and_save(trainer, model, tokenizer, label_encoder, params)
        mlflow.log_metrics(metrics)

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="distilbert_model",
            registered_model_name="callcenter_distilbert_model",
            task="text-classification",
        )

        print("\n✅ ENTRAÎNEMENT TRANSFORMER TERMINÉ!")
        print("🔬 MLflow UI: http://localhost:5000")
        print(f"📊 Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
