"""
Script pour télécharger le dataset IT Service Ticket Classification
"""
import os
from pathlib import Path

import pandas as pd


def download_kaggle_dataset():
    """
    Télécharge le dataset depuis Kaggle
    Vous devez avoir kaggle CLI configuré ou télécharger manuellement
    """

    # Créer le répertoire data/raw s'il n'existe pas
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Téléchargement du dataset IT Service Ticket Classification...")

    # Option 1: Avec Kaggle CLI (si configuré)
    try:
        os.system(
            "kaggle datasets download -d adisongoh/it-service-ticket-classification-dataset -p data/raw --unzip"
        )
        print("✅ Dataset téléchargé avec Kaggle CLI")
    except Exception:
        print("⚠️ Kaggle CLI non configuré")
        print("📋 Étapes manuelles :")
        print(
            "1. Aller sur : https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset"
        )
        print("2. Télécharger le CSV")
        print("3. Placer le fichier dans data/raw/")
        return False

    # Vérifier le fichier téléchargé
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        csv_file = csv_files[0]

        # Charger et examiner les données
        df = pd.read_csv(csv_file)
        print(f"\n📊 Dataset chargé : {csv_file.name}")
        print(f"   - Lignes : {len(df):,}")
        print(f"   - Colonnes : {list(df.columns)}")
        print(f"   - Catégories : {df['Topic_group'].nunique()}")
        print("\n🏷️ Distribution des catégories :")
        print(df["Topic_group"].value_counts())

        # Renommer le fichier de manière standardisée
        standard_name = data_dir / "service_tickets.csv"
        if csv_file.name != "service_tickets.csv":
            csv_file.rename(standard_name)
            print(f"✅ Fichier renommé : {standard_name}")

        return True
    else:
        print("❌ Aucun fichier CSV trouvé")
        return False


if __name__ == "__main__":
    download_kaggle_dataset()
