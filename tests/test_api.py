
from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    result = response.json()

    assert "message" in result
    assert result["documentation"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert "status" in result
    assert "model_loaded" in result
    assert result["status"] == "ok"


def test_api_documentation_is_available():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "paths" in schema
    assert "/predict" in schema["paths"]


def test_prediction_rejects_invalid_store():
    response = client.post(
        "/predict",
        json={
            "store": 0,
            "date": "2015-06-30",
        },
    )

    assert response.status_code == 422


def test_prediction_rejects_invalid_date_format():
    response = client.post(
        "/predict",
        json={
            "store": 1,
            "date": "30-06-2015",
        },
    )

    assert response.status_code == 422


def test_prediction_requires_store():
    response = client.post(
        "/predict",
        json={
            "date": "2015-06-30",
        },
    )

    assert response.status_code == 422


def test_prediction_requires_date():
    response = client.post(
        "/predict",
        json={
            "store": 1,
        },
    )

    assert response.status_code == 422


def test_prediction_rejects_malformed_json_fields():
    response = client.post(
        "/predict",
        json={
            "store": "not-a-number",
            "date": "2015-06-30",
        },
    )

    assert response.status_code == 422