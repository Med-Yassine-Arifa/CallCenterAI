import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional: accent removal
try:
    from unidecode import unidecode
except Exception:
    unidecode = None

# Optional: spaCy for lemmatization
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    """Charger les paramètres depuis params.yaml (avec valeurs par défaut si absent)."""
    defaults = {
        "prepare": {
            "min_text_length": 10,
            "train_size": 0.8,
            "random_state": 42,
            "remove_stopwords": False,
            "use_lemmatization": False,
        }
    }
    cfg_path = Path(path)
    params = defaults
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            try:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    params.update(loaded)
            except Exception:
                pass  # Ignore malformed YAML
    return params["prepare"]


def clean_text(text: str, remove_accents: bool = True) -> str:
    """
    Nettoyer le texte d'un ticket :
    - lowercasing
    - remove URLs / emails / ticket ids
    - keep letters/numbers/spaces
    - collapse whitespace
    """
    if pd.isna(text):
        return ""

    text = str(text)

    if remove_accents and unidecode is not None:
        text = unidecode(text)

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b[a-zA-Z]{1,6}-\d+\b", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_normalize(texts: pd.Series, remove_stopwords: bool = False, use_lemmatization: bool = False) -> pd.Series:
    """
    Tokenize and optionally remove stopwords / lemmatize.
    Returns a joined string (ready for vectorizers).
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set()
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = set()

    def process(doc: str) -> str:
        if not doc:
            return ""
        tokens = word_tokenize(doc)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        if use_lemmatization and nlp is not None:
            doc_sp = nlp(" ".join(tokens))
            tokens = [tok.lemma_ for tok in doc_sp]
        return " ".join(tokens)

    return texts.apply(process)


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyser le dataset et retourner des statistiques basiques."""
    text_lengths = df["Document"].astype(str).apply(len)
    return {
        "total_samples": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "unique_categories": int(df["Topic_group"].nunique()) if "Topic_group" in df.columns else 0,
        "category_distribution": df["Topic_group"].value_counts().to_dict() if "Topic_group" in df.columns else {},
        "text_length_stats": {
            "mean": float(text_lengths.mean()),
            "median": float(text_lengths.median()),
            "std": float(text_lengths.std()),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
        },
    }


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, Dict[str, Any]]:
    """
    Préparer les données pour l'entraînement.
    Retourne :
      train_df, test_df, label_encoder, stats
    """
    params = load_params()
    print("Chargement des données...")

    data_path = Path("data/raw/service_tickets.csv")
    if not data_path.exists():
        possible_files = list(Path("data/raw").glob("*.csv"))
        if possible_files:
            data_path = possible_files[0]
            print(f"⚠  Utilisation du fichier : {data_path}")
        else:
            raise FileNotFoundError("Aucun fichier CSV trouvé dans data/raw/")

    df = pd.read_csv(data_path)
    print(f"✅ Dataset chargé : {len(df)} échantillons")

    stats = analyze_dataset(df)
    print("Statistiques initiales calculées.")

    required_columns = ["Document", "Topic_group"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    initial_size = len(df)
    df = df.dropna(subset=["Document", "Topic_group"])
    print(f"Supprimé {initial_size - len(df)} lignes avec valeurs manquantes")

    df["Document_clean"] = df["Document"].apply(lambda t: clean_text(t, remove_accents=True))

    min_length = int(params.get("min_text_length", 10))
    initial_size = len(df)
    df = df[df["Document_clean"].str.len() >= min_length]
    print(f"Supprimé {initial_size - len(df)} textes < {min_length} caractères")

    initial_size = len(df)
    df = df.drop_duplicates(subset=["Document_clean"])
    print(f"Supprimé {initial_size - len(df)} doublons (Document_clean)")

    if params.get("remove_stopwords") or params.get("use_lemmatization"):
        try:
            df["Document_clean"] = tokenize_and_normalize(
                df["Document_clean"],
                remove_stopwords=bool(params.get("remove_stopwords")),
                use_lemmatization=bool(params.get("use_lemmatization")),
            )
            print("Tokenization / stopword removal / lemmatization appliqués.")
        except Exception as e:
            print("⚠  Tokenization/lemmatization failed:", e)

    label_encoder = LabelEncoder()
    df["Topic_encoded"] = label_encoder.fit_transform(df["Topic_group"].astype(str))
    classes = list(label_encoder.classes_)
    print(f"Catégories encodées : {len(classes)}")

    for i, category in enumerate(classes):
        count = int((df["Topic_encoded"] == i).sum())
        print(f"  {i}: {category} ({count} échantillons)")

    test_size = 1.0 - float(params.get("train_size", 0.8))
    random_state = int(params.get("random_state", 42))

    value_counts = df["Topic_encoded"].value_counts()
    rare_classes = (value_counts < 2).sum()
    if rare_classes > 0:
        print(f"⚠  {rare_classes} classes ont <2 échantillons; pas de stratification.")
        stratify_col = None
    else:
        stratify_col = df["Topic_encoded"]

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_col)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    stats["final_samples"] = len(df)
    stats["train_samples"] = len(train_df)
    stats["test_samples"] = len(test_df)
    stats["classes"] = classes
    stats["class_counts"] = df["Topic_group"].value_counts().to_dict()

    return train_df, test_df, label_encoder, stats


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    train_df, test_df, label_encoder, stats = prepare_data()

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    with open("data/processed/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open("metrics/data_preparation.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Préparation des données terminée!")


if __name__ == "__main__":
    main()
