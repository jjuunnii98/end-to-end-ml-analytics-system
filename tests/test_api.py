from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


client = TestClient(app)


VALID_REQUEST_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.85,
    "TotalCharges": 1081.25,
}


MOCK_INFERENCE_RESPONSE = {
    "model_name": "random_forest",
    "threshold": 0.3,
    "artifact_source": "joblib",
    "artifact_paths": {
        "model_path": "artifacts/model/random_forest_model.joblib",
        "preprocessor_path": "artifacts/model/random_forest_preprocessor.joblib",
    },
    "model_summary": {
        "model_class": "RandomForestClassifier",
        "has_predict": True,
        "has_predict_proba": True,
        "n_features_in": 46,
        "classes": [0, 1],
    },
    "prediction_results": {
        "input_shape": [1, 19],
        "transformed_shape": [1, 46],
        "threshold": 0.3,
        "model_labels": [1],
        "probabilities": [0.6693261186117347],
        "predictions": [
            {
                "predicted_label": 1,
                "model_label": 1,
                "churn_probability": 0.6693261186117347,
            }
        ],
    },
}


def test_root_endpoint_returns_ok() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_prediction_payload(monkeypatch) -> None:
    def mock_run_inference_pipeline(*args, **kwargs):
        return MOCK_INFERENCE_RESPONSE

    monkeypatch.setattr("src.api.main.run_inference_pipeline", mock_run_inference_pipeline)

    response = client.post("/predict", json=VALID_REQUEST_PAYLOAD)

    assert response.status_code == 200

    payload = response.json()
    assert payload["model_name"] == "random_forest"
    assert payload["artifact_source"] == "joblib"
    assert payload["input_shape"] == [1, 19]
    assert payload["transformed_shape"] == [1, 46]
    assert payload["threshold"] == 0.3
    assert payload["model_labels"] == [1]
    assert payload["probabilities"] == [0.6693261186117347]
    assert payload["predictions"][0]["predicted_label"] == 1
    assert payload["predictions"][0]["model_label"] == 1


def test_predict_endpoint_returns_422_for_invalid_request() -> None:
    invalid_payload = VALID_REQUEST_PAYLOAD.copy()
    invalid_payload.pop("gender")

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422