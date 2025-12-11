import logging
import time

import pytest
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCompleteFlow:
    """Tests du flux complet de classification"""

    BASE_URLS = {
        "tfidf": "http://localhost:8001",
        "transformer": "http://localhost:8002",
        "agent": "http://localhost:8003",
    }

    @pytest.fixture(scope="class", autouse=True)
    def wait_for_services(self):
        """Attendre que les services soient prêts"""
        for service, url in self.BASE_URLS.items():
            for i in range(30):
                try:
                    response = requests.get(f"{url}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service} service ready")
                        break
                except Exception:
                    if i == 29:
                        pytest.fail(f"{service} not ready")
                    time.sleep(1)

    def test_e2e_simple_classification(self):
        """E2E: Classification simple"""
        logger.info("\n🧪 Test E2E: Simple Classification")

        test_text = "My laptop screen is completely broken"

        # 1. TF-IDF prediction
        response = requests.post(f"{self.BASE_URLS['tfidf']}/predict", json={"text": test_text})
        assert response.status_code == 200
        tfidf_result = response.json()
        logger.info(f"   TF-IDF: {tfidf_result['predicted_class']} ({tfidf_result['confidence']:.3f})")

        # 2. Transformer prediction
        response = requests.post(f"{self.BASE_URLS['transformer']}/predict", json={"text": test_text})
        assert response.status_code == 200
        transformer_result = response.json()
        logger.info(f"   Transformer: {transformer_result['predicted_class']} ({transformer_result['confidence']:.3f})")

        # 3. Agent intelligent routing
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json={"text": test_text})
        assert response.status_code == 200
        agent_result = response.json()
        logger.info(f"   Agent: {agent_result['predicted_class']} (model: {agent_result['model_used']})")

        # Assertions
        assert tfidf_result["predicted_class"] is not None
        assert transformer_result["predicted_class"] is not None
        assert agent_result["predicted_class"] is not None

    def test_e2e_pii_scrubbing(self):
        """E2E: PII Scrubbing"""
        logger.info("\n🧪 Test E2E: PII Scrubbing")

        text_with_pii = "Contact john.doe@example.com or call 555-123-4567"

        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json={"text": text_with_pii})

        assert response.status_code == 200
        result = response.json()

        assert result["pii_detected"] is True
        assert len(result["pii_details"]) > 0

        logger.info(f"   PII détectées: {result['pii_details']}")

    def test_e2e_routing_logic(self):
        """E2E: Test du routage intelligent"""
        logger.info("\n🧪 Test E2E: Intelligent Routing")

        # Court texte → TF-IDF
        short_text = "This printer not working"
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json={"text": short_text})
        assert response.json()["model_used"] == "tfidf"
        logger.info(f"   Short text routed to: {response.json()['model_used']}")

        # Long texte → Transformer
        long_text = " ".join(
            [
                "My laptop has started shutting down randomly whenever "
                "I run multiple applications at the same time. The fan becomes "
                "extremely loud, and sometimes the screen freezes before the device "
                "powers off completely. I already tried reinstalling drivers, "
                "but the issue keeps happening. Please assist."
            ]
        )
        response = requests.post(f"{self.BASE_URLS['agent']}/predict", json={"text": long_text})
        assert response.json()["model_used"] == "distilbert"
        logger.info(f"   Long text routed to: {response.json()['model_used']}")

    def test_e2e_concurrent_requests(self):
        """E2E: Requêtes concurrentes"""
        logger.info("\n🧪 Test E2E: Concurrent Requests")

        import concurrent.futures

        test_cases = [
            "Laptop not starting",
            "Cannot login to account",
            "Need new hardware",
            "Printer jam",
            "Email not working",
        ] * 10  # Répéter 50 fois

        results = []

        def make_request(text):
            try:
                response = requests.post(
                    f"{self.BASE_URLS['agent']}/predict",
                    json={"text": text},
                    timeout=10,
                )
                return response.status_code == 200
            except Exception:
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, text) for text in test_cases]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_rate = sum(results) / len(results) * 100
        logger.info(f"   Success rate: {success_rate:.1f}%")

        assert success_rate >= 95, f"Success rate too low: {success_rate:.1f}%"

    def test_e2e_model_consistency(self):
        """E2E: Consistance des modèles"""
        logger.info("\n🧪 Test E2E: Model Consistency")

        test_text = "Database connection failed"

        # Exécuter 3 fois
        predictions = []
        for i in range(3):
            response = requests.post(f"{self.BASE_URLS['agent']}/predict", json={"text": test_text})
            predictions.append(response.json()["predicted_class"])

        # Tous les résultats doivent être identiques
        assert len(set(predictions)) == 1, "Predictions are not consistent"
        logger.info(f"   All predictions: {predictions[0]}")

    def test_e2e_response_time_sla(self):
        """E2E: Vérifier les SLAs de latence"""
        logger.info("\n🧪 Test E2E: Response Time SLAs")

        test_text = "Hardware issue"

        # SLA targets
        slas = {
            "tfidf": 0.1,  # 100ms
            "transformer": 0.5,  # 500ms
            "agent": 0.15,  # 150ms (après routage)
        }

        for service, url in self.BASE_URLS.items():
            start = time.time()
            response = requests.post(f"{url}/predict", json={"text": test_text})
            logger.info(f"   {response}: {response.status_code}")
            elapsed = time.time() - start

            sla = slas[service]
            status = "✅" if elapsed < sla else "⚠️"
            logger.info(f"   {service}: {elapsed*1000:.0f}ms (SLA: {sla*1000:.0f}ms) {status}")

            # Warning but not failure for SLA
            if elapsed > sla * 1.5:
                logger.warning(f"   {service} significantly exceeds SLA")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
