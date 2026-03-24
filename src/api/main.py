from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    HealthResponse,
)
from src.pipelines.inference_pipeline import run_inference_pipeline


app = FastAPI(
    title="End-to-End ML Analytics System API",
    description="FastAPI service for Telco customer churn prediction using a trained ML pipeline.",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """
    Basic root endpoint for quick service checks.
    """
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.
    """
    return HealthResponse(status="ok")


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest) -> ChurnPredictionResponse:
    """
    Predict churn probability and label for a single customer record.

    The endpoint uses the saved model and preprocessor artifacts via the
    shared inference pipeline.
    """
    try:
        inference_results = run_inference_pipeline(
            input_data=request.model_dump(),
            model_name="random_forest",
            threshold=0.3,
        )
        prediction_results = inference_results["prediction_results"]

        response_payload = {
            "model_name": inference_results["model_name"],
            "artifact_source": inference_results["artifact_source"],
            "input_shape": prediction_results["input_shape"],
            "transformed_shape": prediction_results["transformed_shape"],
            "threshold": prediction_results["threshold"],
            "model_labels": prediction_results["model_labels"],
            "probabilities": prediction_results["probabilities"],
            "predictions": prediction_results["predictions"],
        }

        return ChurnPredictionResponse(**response_payload)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Required model artifacts were not found. "
                "Run the training pipeline first to generate saved model files. "
                f"Details: {exc}"
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error during prediction: {exc}",
        ) from exc