import logging
from collections import defaultdict
from threading import Lock

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collecteur centralisé de métriques métier"""

    def __init__(self):
        self.lock = Lock()

        # =====================================================================
        # Métriques de Prédiction
        # =====================================================================

        # Distribution des prédictions par classe
        self.predictions_by_class = Counter(
            "predictions_by_class_total",
            "Nombre de prédictions par classe",
            ["class_name", "service"],
        )

        # Confiance des prédictions
        self.prediction_confidence = Histogram(
            "prediction_confidence",
            "Distribution de la confiance des prédictions",
            ["service"],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
        )

        # Score des prédictions (F1, Accuracy si disponible)
        self.prediction_score = Gauge(
            "prediction_score",
            "Score de prédiction (F1, Accuracy)",
            ["metric_type", "service"],
        )

        # =====================================================================
        # Métriques de Sécurité & PII
        # =====================================================================

        # Nombre de PII détectées
        self.pii_detected = Counter(
            "pii_detected_total", "Nombre total de PII détectées", ["pii_type"]
        )

        # Distribution des types de PII
        self.pii_by_type = Counter("pii_by_type_total", "PII par type", ["pii_type"])

        # =====================================================================
        # Métriques de Routage
        # =====================================================================

        # Routage des requêtes par modèle
        self.routing_by_model = Counter(
            "routing_by_model_total",
            "Nombre de routages vers chaque modèle",
            ["model", "reason"],
        )

        # Latence par modèle
        self.latency_by_model = Histogram(
            "latency_by_model_seconds",
            "Latence par modèle",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        # =====================================================================
        # Métriques d'Erreurs
        # =====================================================================

        # Erreurs par type
        self.errors_by_type = Counter(
            "errors_by_type_total",
            "Nombre d'erreurs par type",
            ["error_type", "service"],
        )

        # Taux d\'erreurs en temps réel
        self.error_rate = Gauge(
            "error_rate_percent", "Taux d'erreurs en pourcentage", ["service"]
        )

        # =====================================================================
        # Métriques de Performance
        # =====================================================================

        # Throughput (requêtes par seconde)
        self.throughput = Gauge("throughput_rps", "Requêtes par seconde", ["service"])

        # P95 latency
        self.p95_latency = Gauge("p95_latency_seconds", "P95 latency", ["service"])

        # P99 latency
        self.p99_latency = Gauge("p99_latency_seconds", "P99 latency", ["service"])

        # =====================================================================
        # Stockage interne pour calculs statistiques
        # =====================================================================

        self.predictions_buffer = defaultdict(list)  # Pour calculs P95/P99
        self.errors_buffer = defaultdict(int)
        self.total_requests = defaultdict(int)

    def record_prediction(self, service, class_name, confidence, processing_time):
        """Enregistrer une prédiction"""
        with self.lock:
            self.predictions_by_class.labels(
                class_name=class_name, service=service
            ).inc()

            self.prediction_confidence.labels(service=service).observe(confidence)

            # Stocker pour calculs P95/P99
            self.predictions_buffer[service].append(
                {"time": processing_time, "confidence": confidence, "class": class_name}
            )

            self.total_requests[service] += 1

    def record_pii(self, pii_types):
        """Enregistrer les PII détectées"""
        with self.lock:
            self.pii_detected.inc(len(pii_types))

            for pii_type in pii_types:
                self.pii_by_type.labels(pii_type=pii_type).inc()

    def record_routing(self, model, reason):
        """Enregistrer le routage"""
        with self.lock:
            self.routing_by_model.labels(model=model, reason=reason).inc()

    def record_latency(self, model, latency):
        """Enregistrer la latence"""
        with self.lock:
            self.latency_by_model.labels(model=model).observe(latency)

    def record_error(self, service, error_type):
        """Enregistrer une erreur"""
        with self.lock:
            self.errors_by_type.labels(error_type=error_type, service=service).inc()
            self.errors_buffer[service] += 1

    def update_statistics(self, service):
        """Mettre à jour les statistiques calculées"""
        with self.lock:
            # Vérifier qu'il y a des données
            if (
                service not in self.predictions_buffer
                or not self.predictions_buffer[service]
            ):
                return

            # Extraire les latences
            latencies = [p["time"] for p in self.predictions_buffer[service]]

            # Calculer P95 et P99
            if latencies:
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)

                self.p95_latency.labels(service=service).set(p95)
                self.p99_latency.labels(service=service).set(p99)

            # Calculer le taux d'erreurs
            total = self.total_requests[service]
            if total > 0:
                error_rate = (self.errors_buffer[service] / total) * 100
                self.error_rate.labels(service=service).set(error_rate)

            # Nettoyer le buffer (pour éviter overflow mémoire)
            if len(self.predictions_buffer[service]) > 1000:
                self.predictions_buffer[service] = self.predictions_buffer[service][
                    -500:
                ]


# Instance globale
metrics_collector = MetricsCollector()
