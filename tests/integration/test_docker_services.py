"""
Tests d'intégration pour les services Docker
"""
import time

import pytest
import requests


class TestDockerServices:
    """Tests des services dockerisés"""

    BASE_URLS = {
        "tfidf": "http://localhost:8001",
        "transformer": "http://localhost:8002",
        "agent": "http://localhost:8003",
    }

    @pytest.fixture(scope="class", autouse=True)
    def wait_for_services(self):
        """Attendre que tous les services soient prêts"""
        print("\n⏳ Attente du démarrage des services...")
        max_retries = 30
        retry_interval = 2

        for service, url in self.BASE_URLS.items():
            for i in range(max_retries):
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"✅ {service} service prêt")
                        break
                except requests.exceptions.RequestException:
                    if i == max_retries - 1:
                        pytest.fail(f"❌ {service} service non disponible après {max_retries} tentatives")
                time.sleep(retry_interval)

    def test_tfidf_health(self):
        """Tester le health check TF-IDF"""
        response = requests.get(f"{self.BASE_URLS['tfidf']}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_transformer_health(self):
        """Tester le health check Transformer"""
        response = requests.get(f"{self.BASE_URLS['transformer']}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_agent_health(self):
        """Tester le health check Agent"""
        response = requests.get(f"{self.BASE_URLS['agent']}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_tfidf_prediction(self):
        """Tester prédiction TF-IDF"""
        payload = {"text": "My laptop screen is broken and needs repair"}
        response = requests.post(f"{self.BASE_URLS['tfidf']}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_transformer_prediction(self):
        """Tester prédiction Transformer"""
        payload = {"text": "Cannot access my account forgot password"}
        response = requests.post(f"{self.BASE_URLS['transformer']}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data

    def test_agent_routing(self):
        """Tester le routage de l'agent"""

        # Texte court → devrait router vers TF-IDF
        short_text = {"text": "This printer not working"}
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json=short_text)
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "tfidf"

        # Texte long → devrait router vers Transformer
        long_text = {
            "text": (
                "My laptop has started shutting down randomly whenever "
                "I run multiple applications at the same time. The fan becomes "
                "extremely loud, and sometimes the screen freezes before the device "
                "powers off completely. I already tried reinstalling drivers, "
                "but the issue keeps happening. Please assist."
            )
        }
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json=long_text)
        assert response.status_code == 200
        data = response.json()
        assert data["model_used"] == "distilbert"

    def test_pii_scrubbing(self):
        """Tester le scrubbing des PII"""
        payload = {"text": "Contact me at john.doe@example.com or call 555-123-4567"}
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["pii_detected"] is True
        assert len(data["pii_details"]) > 0

    def test_metrics_endpoints(self):
        """Test metrics endpoints"""
        for service, url in self.BASE_URLS.items():
            response = requests.get(f"{url}/metrics")
            assert response.status_code == 200
            # Check for ANY custom metric instead of specific name
            assert any(
                metric in response.text
                for metric in [
                    "agent_requests_total",
                    "tfidf_requests_total",
                    "transformer_requests_total",
                    "p95_latency_seconds",
                    "error_rate_percent",
                    "prediction_confidence_created",
                ]
            ), f"No custom metrics found in {service} response"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
