# CallCenter MLOps

## 🎯 Objectif

Solution MLOps complète pour classifier automatiquement les tickets clients d'un centre d'appel en utilisant deux approches NLP :

* **Modèle Classique** : TF-IDF + SVM
* **Modèle Avancé** : Transformer (DistilBERT multilingue)

---

## 🎗️ Architecture

```
callcenter-mlops/
├── data/                    # Données du projet
│   ├── raw/                 # Données brutes (CSV Kaggle)
│   └── processed/           # Données préprocessées
├── src/                     # Code source
│   ├── data_preparation/    # Scripts de préparation des données
│   ├── tfidf_service/       # Service API TF-IDF
│   ├── transformer_service/ # Service API Transformer
│   └── agent_service/       # Agent IA orchestrateur
├── models/                  # Modèles ML sauvegardés
├── tests/                   # Tests unitaires et d'intégration
├── monitoring/              # Configuration monitoring (Prometheus/Grafana)
├── docker/                  # Dockerfiles pour chaque service
└── scripts/                 # Scripts utilitaires
```

---

## 🚀 Installation

### Prérequis

* Python 3.11.9
* Git
* Docker & Docker Compose (pour la semaine 2+)

### Configuration de l'environnement

1. Cloner le projet

```bash
git clone https://github.com/Med-Yassine-Arifa/CallCenterAI
cd callcenter-mlops
```

2. Créer un environnement virtuel

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Installer les dépendances

```bash
pip install -r requirements.txt
```

4. Configurer pre-commit

```bash
pre-commit install
```

5. Télécharger les données

```bash
python scripts/download_data.py
```

6. Configurer MLflow

```bash
python -c "from mlflow_configs.mlflow_config import setup_mlflow; setup_mlflow()"
```

---

## 📊 Dataset

* **Source** : [Kaggle IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)
* **Taille** : 47,837 tickets
* **Colonnes** :

  * `Document` : Texte du ticket
  * `Topic_group` : Catégorie (8 classes)

---

## 🛠️ Utilisation

### Démarrer MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///{MLFLOW_TRACKING_URI}/mlflow.db --default-artifact-root C:/CallCenterAI/mlruns --host 0.0.0.0 --port 5000
```

Interface disponible : [http://localhost:5000](http://localhost:5000)

### Exécuter le pipeline DVC

```bash
dvc repro
```

### Lancer les tests

```bash
python -m pytest tests/ -v --cov=src
```

---

## 📋 Pipeline MLOps

1. **prepare_data** : Préprocessing et split train/test
2. **train_tfidf** : Entraînement modèle TF-IDF + SVM
3. **train_transformer** : Fine-tuning DistilBERT
4. **evaluate_models** : Comparaison des performances

---

## 🧪 Tests

* Tests unitaires :

```bash
python -m pytest tests/unit/ -v
```

* Tests d'intégration :

```bash
python -m pytest tests/integration/ -v
```

* Coverage :

```bash
python -m pytest --cov=src --cov-report=html
```

---

## 📊 Monitoring

* **MLflow** : Tracking des expériences et modèles
* **DVC** : Versioning des données et pipeline
* **Pre-commit** : Qualité du code

---

## 🤝 Contribution

1. Créer une branche :

```bash
git checkout -b feature/ma-fonctionnalite
```

2. Committer :

```bash
git commit -am 'Ajouter ma fonctionnalité'
```

3. Pusher :

```bash
git push origin feature/ma-fonctionnalite
```

4. Créer une Pull Request

---

## 📚 Guide de Développement

Voir `docs/DEVELOPMENT.md` pour les détails sur :

* MLflow : Tracking des expériences
* DVC : Versioning et pipeline
* Standards de code : black, isort, flake8, bandit
