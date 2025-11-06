"""
Script pour tester le service TF-IDF localement
"""


from pprint import pprint

import requests

BASE_URL = "http://localhost:8001"


def test_root():
    """Tester l'endpoint racine"""
    print("\n" + "=" * 60)
    print("TEST: Endpoint racine")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_health():
    """Tester le health check"""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_prediction():
    """Tester une prédiction"""
    print("\n" + "=" * 60)
    print("TEST: Prédiction")
    print("=" * 60)

    test_cases = [
        "My computer screen is broken and not displaying anything",
        "I forgot my password and cannot login to the system",
        "Need to purchase new laptop for new employee",
        "Printer is not working showing paper jam error",
        "Email server is down and I cannot send messages",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Text: {text}")

        response = requests.post(f"{BASE_URL}/predict", json={"text": text})

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Catégorie: {result['predicted_class']}")
            print(f"   Confiance: {result['confidence']:.3f}")
            print(f"   Temps: {result['processing_time_ms']:.2f}ms")
            print("   Top 3 probas:")

            for cat, prob in list(result["probabilities"].items())[:3]:
                print(f"     {cat}: {prob:.3f}")
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(response.text)


def test_model_info():
    """Tester les infos du modèle"""
    print("\n" + "=" * 60)
    print("TEST: Informations Modèle")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    pprint(response.json())


def test_metrics():
    """Tester les métriques Prometheus"""
    print("\n" + "=" * 60)
    print("TEST: Métriques Prometheus")
    print("=" * 60)

    response = requests.get(f"{BASE_URL}/metrics")
    print(f"Status: {response.status_code}")
    print("Échantillon des métriques:")
    print(response.text[:500] + "...")


if __name__ == "__main__":
    print("🧪 Tests du Service TF-IDF")
    print("Assurez-vous que le service est lancé sur le port 8001\n")

    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction()
        test_metrics()

        print("\n" + "=" * 60)
        print("✅ Tous les tests sont terminés!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Erreur: Impossible de se connecter au service")
        print("Lancez d'abord: cd src/tfidf_service && python -m api.main")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
