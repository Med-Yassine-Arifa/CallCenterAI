import pickle
import pandas as pd
from pathlib import Path
def test_tfidf_model():
    """Tester le modèle TF-IDF avec des exemples"""
    
    print(" Test du modèle TF-IDF...")
    
    # Charger le modèle
    with open("models/tfidf_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Charger l'encodeur
    with open("data/processed/label_encoder.pkl", 'rb') as f:
     label_encoder = pickle.load(f)
    
    print(f"✅ Modèle chargé")
    print(f"   Classes disponibles: {len(label_encoder.classes_)}")
    
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