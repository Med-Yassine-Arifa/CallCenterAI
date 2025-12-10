try:
    import joblib
except ImportError:
    import pickle as joblib  # Fallback if joblib not available


def test_tfidf_model():
    """Tester le modèle TF-IDF avec des exemples"""

    print(" Test du modèle TF-IDF...")

    # Charger le modÞle
    model = joblib.load("models/tfidf_model.pkl")  # noqa: B301

    # Charger l'encodeur
    label_encoder = joblib.load("data/processed/label_encoder.pkl")  # noqa: B301

    print("✅ Modèle chargé")
    print("   Classes disponibles: {len(label_encoder.classes_)}")

    # Exemples de test
    test_texts = [
        "my computer is not starting up and showing blue screen",
        "i forgot my password and cannot access my account",
        "the printer is not working and paper is jammed",
        "need to order new laptop for the new employee",
        "email server is down and cannot send messages",
    ]

    print("\n Prédictions sur exemples:")
    print("-" * 80)

    for i, text in enumerate(test_texts, 1):
        # Prédiction
        prediction = model.predict([text])[0]
        probas = model.predict_proba([text])[0]

        # Décoder la catégorie
        category = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probas)

        print(f"{i}. Texte: {text}")
        print(f"   Catégorie: {category}")
        print(f"   Confiance: {confidence:.3f}")
        print()


if __name__ == "__main__":
    test_tfidf_model()
