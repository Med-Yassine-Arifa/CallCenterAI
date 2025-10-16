"""
Script pour tester le modèle Transformer entraîné
"""
import pickle
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_transformer_model():
    """Tester le modèle Transformer avec des exemples"""
    print("🧪 Test du modèle Transformer...")

    # Vérifier si le modèle existe
    model_path = Path("models/transformer_model")
    if not model_path.exists():
        print("❌ Modèle Transformer non trouvé. Lancez d'abord l'entraînement.")
        return

    # Charger le modèle et tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Charger l'encodeur de labels
    with open("data/processed/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("✅ Modèle chargé")
    print(f"Classes disponibles: {len(label_encoder.classes_)}")

    # Exemples de test
    test_texts = [
        "my laptop screen is broken and needs repair",
        "forgot my login credentials and cannot access system",
        "printer not responding and showing error message",
        "request new equipment for employee onboarding",
        "email service down and cannot receive messages",
    ]

    print("\n🔍 Prédictions sur exemples:")
    print("-" * 80)

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(test_texts, 1):
            # Tokenizer le texte
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            # Prédiction
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Récupérer la prédiction
            predicted_class_id = outputs.logits.argmax(dim=-1).item()
            confidence = probabilities[0][predicted_class_id].item()

            # Décoder la catégorie
            category = label_encoder.classes_[predicted_class_id]

            print(f"{i}. Texte: {text}")
            print(f"   Catégorie: {category}")
            print(f"   Confiance: {confidence:.3f}\n")


if __name__ == "__main__":
    test_transformer_model()
