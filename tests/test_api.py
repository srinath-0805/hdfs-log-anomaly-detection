
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from api.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_normal_prediction():
    payload = {
        "event_ids": [
            5, 22, 5, 5, 11, 9, 11, 9, 11, 9,
            26, 26, 26, 4, 3, 4, 2, 23, 23, 23, 21, 21
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == "NORMAL"


def test_anomaly_prediction():
    payload = {
        "event_ids": [
            5, 5, 5, 22, 5, 11, 10, 14, 7, 8,
            11, 15, 5, 11, 9, 26, 14, 7, 11, 10,
            12, 14, 7, 5, 19, 14, 7, 11, 10, 14,
            7, 8, 11, 25, 18, 5, 5, 6, 16, 6,
            26, 26, 23, 23, 23, 21, 21, 21, 20
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == "ANOMALY"


def test_invalid_event_ids():
    payload = {
        "event_ids": [5, 22, 0, 35, 11]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid event ID in sequence."
