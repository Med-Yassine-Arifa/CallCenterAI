"""
Analyse approfondie des features et de leur impact sur les prédictions
Semaine 3 - Feature Engineering
"""
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class FeatureAnalyzer:
    """Analyser l'impact des features sur les prédictions"""

    def __init__(self):
        self.train_df = pd.read_csv("data/processed/train.csv")
        self.vectorizer = None
        self.feature_names = None

    def analyze_text_characteristics(self):
        """Analyser les caractéristiques du texte par classe"""
        print("\n" + "=" * 60)
        print("ANALYSE DES CARACTÉRISTIQUES DU TEXTE")
        print("=" * 60)

        # Créer features de texte
        self.train_df["word_count"] = (
            self.train_df["Document_clean"].str.split().str.len()
        )
        self.train_df["char_count"] = self.train_df["Document_clean"].str.len()
        self.train_df["avg_word_length"] = self.train_df["Document_clean"].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )
        self.train_df["unique_words"] = self.train_df["Document_clean"].apply(
            lambda x: len(set(x.split()))
        )
        self.train_df["unique_ratio"] = (
            self.train_df["unique_words"] / self.train_df["word_count"]
        ).fillna(0)

        # Charger label encoder
        with open("data/processed/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Map des labels
        self.train_df["Topic"] = self.train_df["Topic_encoded"].map(
            {i: label for i, label in enumerate(label_encoder.classes_)}
        )

        # Statistiques par classe
        stats_by_class = self.train_df.groupby("Topic")[
            ["word_count", "char_count", "avg_word_length", "unique_ratio"]
        ].describe()

        print("\n📊 Statistiques par classe:")
        print(stats_by_class)

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Analyse des Caractéristiques Texte par Classe", fontsize=16)

        # Word count
        self.train_df.boxplot(column="word_count", by="Topic", ax=axes[0, 0])
        axes[0, 0].set_title("Distribution du nombre de mots")
        axes[0, 0].set_xlabel("Classe")
        axes[0, 0].set_ylabel("Nombre de mots")

        # Char count
        self.train_df.boxplot(column="char_count", by="Topic", ax=axes[0, 1])
        axes[0, 1].set_title("Distribution du nombre de caractères")
        axes[0, 1].set_xlabel("Classe")
        axes[0, 1].set_ylabel("Nombre de caractères")

        # Avg word length
        self.train_df.boxplot(column="avg_word_length", by="Topic", ax=axes[1, 0])
        axes[1, 0].set_title("Longueur moyenne des mots")
        axes[1, 0].set_xlabel("Classe")
        axes[1, 0].set_ylabel("Longueur")

        # Unique ratio
        self.train_df.boxplot(column="unique_ratio", by="Topic", ax=axes[1, 1])
        axes[1, 1].set_title("Ratio de mots uniques")
        axes[1, 1].set_xlabel("Classe")
        axes[1, 1].set_ylabel("Ratio")

        plt.tight_layout()
        plots_dir = Path("notebooks/plots")
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / "feature_analysis.png", dpi=150, bbox_inches="tight")
        print("\n✅ Graphique sauvegardé: notebooks/plots/feature_analysis.png")

    def extract_and_analyze_tfidf_features(self):
        """Extraire et analyser les features TF-IDF"""
        print("\n" + "=" * 60)
        print("ANALYSE DES FEATURES TF-IDF PAR CLASSE")
        print("=" * 60)

        # Charger label encoder
        with open("data/processed/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        # Map des labels
        self.train_df["Topic"] = self.train_df["Topic_encoded"].map(
            {i: label for i, label in enumerate(label_encoder.classes_)}
        )

        # Extraire TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8
        )

        X_tfidf = vectorizer.fit_transform(self.train_df["Document_clean"])
        self.feature_names = vectorizer.get_feature_names_out()

        # Top features par classe
        print("\n🔑 Top 10 features par classe:")

        top_features_by_class = {}

        for class_id, class_name in enumerate(label_encoder.classes_):
            class_mask = (self.train_df["Topic_encoded"] == class_id).values

            if class_mask.sum() > 0:
                class_X_tfidf = X_tfidf[class_mask]

                # Mean TF-IDF scores
                mean_scores = np.asarray(class_X_tfidf.mean(axis=0)).flatten()

                # Top indices
                top_indices = np.argsort(mean_scores)[::-1][:10]

                top_features = [
                    (self.feature_names[i], mean_scores[i]) for i in top_indices
                ]

                top_features_by_class[class_name] = top_features

                print(f"\n{class_name}:")
                for i, (feature, score) in enumerate(top_features, 1):
                    print(f"  {i}. {feature}: {score:.4f}")

        return top_features_by_class

    def analyze_class_confusion(self):
        """Analyser les classes qui se confondent"""
        print("\n" + "=" * 60)
        print("ANALYSE DE LA CONFUSION ENTRE CLASSES")
        print("=" * 60)

        # Charger les métriques TF-IDF si disponibles
        metrics_file = Path("metrics/tfidf_metrics.json")
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                conf_matrix = np.array(metrics["confusion_matrix"])
                classes = metrics["classes"]

                print("\n📊 Matrice de confusion normalisée:")

                # Normaliser pour voir les pourcentages
                conf_matrix_norm = (
                    conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
                )

                # Afficher confusion pour chaque classe
                for i, class_name in enumerate(classes):
                    print(f"\n{class_name}:")
                    for j, pred_name in enumerate(classes):
                        percentage = conf_matrix_norm[i][j] * 100
                        if percentage > 5:  # Afficher seulement > 5%
                            print(f"  → {pred_name}: {percentage:.1f}%")


def main():
    """Exécuter l'analyse complète"""
    analyzer = FeatureAnalyzer()

    # Analyser les caractéristiques texte
    analyzer.analyze_text_characteristics()

    # Analyser les confusions
    analyzer.analyze_class_confusion()

    print("\n✅ ANALYSE COMPLÈTE TERMINÉE!")


if __name__ == "__main__":
    main()
