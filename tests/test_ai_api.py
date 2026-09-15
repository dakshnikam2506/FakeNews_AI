"""
FakeNews_AI - AI API Integration Test

Tests:
    GET  /
    GET  /health
    POST /predict
"""

from fastapi.testclient import TestClient

from src.ai_api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "FakeNews_AI"
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "FakeNews_AI"


def test_predict():
    claim = "The government announced a new industrial policy."

    response = client.post(
        "/predict",
        json={"claim": claim}
    )

    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert data["claim"] == claim
    assert data["decision"] in {
        "REAL",
        "FAKE",
        "UNCERTAIN"
    }

    assert isinstance(data["reason"], str)

    # ML
    assert data["ml"]["prediction"] in {
        "REAL",
        "FAKE"
    }

    assert 0.0 <= data["ml"]["confidence"] <= 1.0
    assert 0.0 <= data["ml"]["model_agreement"] <= 1.0

    # Evidence
    assert data["evidence"]["signal"] in {
        "STRONG",
        "MODERATE",
        "WEAK",
        "INSUFFICIENT"
    }

    assert data["evidence"]["stance"] in {
        "SUPPORTING",
        "CONTRADICTING",
        "RELATED",
        "INSUFFICIENT"
    }

    assert isinstance(
        data["evidence"]["articles_evaluated"],
        int
    )

    # NLI
    assert "entailment" in data["nli"]
    assert "contradiction" in data["nli"]
    assert "neutral" in data["nli"]

    assert isinstance(data["nli"]["entailment"], int)
    assert isinstance(data["nli"]["contradiction"], int)
    assert isinstance(data["nli"]["neutral"], int)

    # Articles
    assert isinstance(data["articles"], list)

    # XAI
    assert "xai" in data

    assert isinstance(
        data["xai"]["available"],
        bool
    )

    assert data["xai"]["model"] == "Logistic Regression"

    # Error should normally be null
    assert "error" in data


def test_empty_claim():
    response = client.post(
        "/predict",
        json={"claim": "   "}
    )

    assert response.status_code == 400


def test_missing_claim():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


if __name__ == "__main__":
    print("=" * 70)
    print("DAY 23 — AI API INTEGRATION TEST")
    print("=" * 70)

    tests = [
        test_root,
        test_health,
        test_predict,
        test_empty_claim,
        test_missing_claim,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"[PASS] {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}")
            print(f"       {e}")
            failed += 1

    print()
    print("-" * 70)
    print(f"TOTAL : {len(tests)}")
    print(f"PASS  : {passed}")
    print(f"FAIL  : {failed}")
    print("-" * 70)

    if failed == 0:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")