
from app import app


def test_health():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_prediction():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]},
    )

    assert response.status_code == 200
    assert response.json["species"] == "setosa"


def test_invalid_prediction():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5]},
    )

    assert response.status_code == 400

