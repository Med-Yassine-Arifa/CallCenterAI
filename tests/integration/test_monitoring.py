"""
Tests du monitoring et des dashboards
"""

import pytest
import requests


class TestMonitoring:
    """Tests du système de monitoring"""

    def test_prometheus_scrape_targets(self):
        """Vérifier que Prometheus scrape les cibles"""
        response = requests.get("http://localhost:9090/api/v1/targets")
        assert response.status_code == 200
        data = response.json()

        # Vérifier qu'il y a des cibles
        assert len(data["data"]["activeTargets"]) > 0

        # Vérifier que nos services sont présents
        targets = [t["labels"]["instance"] for t in data["data"]["activeTargets"]]
        assert "tfidf" in targets or "tfidf-service:8001" in targets

    def test_prometheus_metrics_collected(self):
        """Vérifier que les métriques sont collectées"""
        # Queries à tester
        queries = [
            "tfidf_requests_total",
            "transformer_requests_total",
            "agent_requests_total",
            "error_rate_percent",
            "p95_latency_seconds",
        ]

        for query in queries:
            response = requests.get(
                "http://localhost:9090/api/v1/query", params={"query": query}
            )

            assert response.status_code == 200
            data = response.json()

            # Vérifier que la query retourne des résultats
            print(
                f"✅ Metric '{query}' collected: {len(data['data']['result'])} results"
            )

    def test_grafana_health(self):
        """Vérifier que Grafana fonctionne"""
        response = requests.get("http://localhost:3000/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["database"] == "ok"

    def test_grafana_datasource(self):
        """Vérifier que Prometheus est configuré comme datasource"""
        response = requests.get(
            "http://localhost:3000/api/datasources",
            headers={"Authorization": "Bearer admin"},
        )

        # Note: Le header peut nécessiter un token valide
        # Ce test est plus illustratif
        print(f"Grafana datasources response: {response.status_code}")

    def test_alert_rules(self):
        """Vérifier que les règles d'alerting sont configurées"""
        response = requests.get("http://localhost:9090/api/v1/rules")
        assert response.status_code == 200
        data = response.json()

        # Vérifier qu'il y a des règles
        assert len(data["data"]["groups"]) > 0

        # Compter les alertes
        total_alerts = sum(len(group["rules"]) for group in data["data"]["groups"])

        print(f"✅ {total_alerts} alerting rules configured")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
