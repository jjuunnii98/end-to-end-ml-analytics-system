from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from src.models.predict import run_inference


DEFAULT_MODEL_NAME = "random_forest"
DEFAULT_THRESHOLD = 0.3
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MODEL_DIR = DEFAULT_ARTIFACTS_DIR / "model"


def _validate_inference_pipeline_inputs(
    model_name: str,
    threshold: float,
) -> None:
    """
    Validate user-provided inference pipeline inputs.

    Parameters
    ----------
    model_name : str
        Model name requested for training before inference.
    threshold : float
        Threshold used for binary prediction.

    Raises
    ------
    ValueError
        If the threshold is outside [0, 1].
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("`model_name` must be a non-empty string.")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be between 0 and 1. Received: {threshold}")


def _build_inference_metadata(
    model_name: str,
    threshold: float,
    data_path: str | Path | None,
) -> dict[str, Any]:
    """
    Build lightweight metadata for the current inference pipeline run.

    Parameters
    ----------
    model_name : str
        Selected model name.
    threshold : float
        Threshold used for binary prediction.
    data_path : str | Path | None
        Optional custom data path used during training.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary describing the current inference run.
    """
    resolved_data_path = str(data_path) if data_path is not None else "default_project_path"

    return {
        "model_name": model_name,
        "threshold": threshold,
        "data_path": resolved_data_path,
    }


def _resolve_artifact_paths(
    model_name: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Path]:
    """
    Resolve the saved model and preprocessor artifact paths.

    Parameters
    ----------
    model_name : str
        Model name used for artifact naming.
    model_dir : Path, default=artifacts/model
        Directory containing serialized model artifacts.

    Returns
    -------
    dict[str, Path]
        Dictionary containing resolved artifact paths.
    """
    return {
        "model_path": model_dir / f"{model_name}_model.joblib",
        "preprocessor_path": model_dir / f"{model_name}_preprocessor.joblib",
    }



def load_inference_artifacts(
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[Any, Any, dict[str, str]]:
    """
    Load a serialized model and preprocessor for inference.

    Parameters
    ----------
    model_name : str, default="random_forest"
        Model name used for artifact naming.
    model_dir : Path, default=artifacts/model
        Directory containing serialized model artifacts.

    Returns
    -------
    tuple[Any, Any, dict[str, str]]
        Loaded model, loaded preprocessor, and artifact path metadata.

    Raises
    ------
    FileNotFoundError
        If the required artifact files do not exist.
    """
    artifact_paths = _resolve_artifact_paths(model_name=model_name, model_dir=model_dir)
    model_path = artifact_paths["model_path"]
    preprocessor_path = artifact_paths["preprocessor_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    return model, preprocessor, {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }



def _summarize_loaded_model(model) -> dict[str, Any]:
    """
    Build a lightweight summary of a loaded model for debugging and API responses.
    """
    summary: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "has_predict": hasattr(model, "predict"),
        "has_predict_proba": hasattr(model, "predict_proba"),
    }

    if hasattr(model, "n_features_in_"):
        summary["n_features_in"] = int(model.n_features_in_)

    if hasattr(model, "classes_"):
        try:
            summary["classes"] = [int(x) for x in model.classes_]
        except Exception:
            summary["classes"] = list(model.classes_)

    return summary


def run_inference_pipeline(
    input_data,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = DEFAULT_THRESHOLD,
    data_path: str | Path | None = None,
    model: Any | None = None,
    preprocessor: Any | None = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> dict[str, Any]:
    """
    Run the full inference pipeline for one or more raw input records.

    Current workflow
    ----------------
    1. Load serialized model and preprocessor artifacts, or use provided preloaded objects
    2. Reuse the shared prediction workflow for schema alignment, transformation,
       probability generation, and threshold-based prediction
    3. Return inference results together with inference metadata and artifact context

    Parameters
    ----------
    input_data : dict | pandas.DataFrame
        Raw inference input.
    model_name : str, default="random_forest"
        Model name used to train the temporary inference model.
    threshold : float, default=0.3
        Decision threshold for binary classification.
    data_path : str | Path | None, default=None
        Optional custom path to the Telco CSV file.
    model : object | None
        Optional preloaded model to avoid retraining.
    preprocessor : object | None
        Optional preloaded preprocessor to avoid retraining.
    model_dir : Path, default=artifacts/model
        Directory containing serialized model and preprocessor artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary containing inference metadata, training context,
        and prediction outputs.
    """
    _validate_inference_pipeline_inputs(model_name=model_name, threshold=threshold)
    # Note: In production, replace prints with a structured logger
    inference_metadata = _build_inference_metadata(
        model_name=model_name,
        threshold=threshold,
        data_path=data_path,
    )

    if model is None or preprocessor is None:
        model, preprocessor, artifact_paths = load_inference_artifacts(
            model_name=model_name,
            model_dir=model_dir,
        )
        model_summary = _summarize_loaded_model(model)
        model_context = {
            "artifact_source": "joblib",
            "artifact_paths": artifact_paths,
        }
    else:
        model_summary = _summarize_loaded_model(model)
        model_context = {
            "artifact_source": "external_objects",
            "artifact_paths": {},
        }

    prediction_results = run_inference(
        model=model,
        preprocessor=preprocessor,
        input_data=input_data,
        threshold=threshold,
    )

    return {
        "model_name": inference_metadata["model_name"],
        "threshold": inference_metadata["threshold"],
        "artifact_source": model_context["artifact_source"],
        "artifact_paths": model_context["artifact_paths"],
        "model_summary": model_summary,
        "prediction_results": prediction_results,
    }


if __name__ == "__main__":
    sample_input = {
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

    inference_results = run_inference_pipeline(
        input_data=sample_input,
        model_name="random_forest",
        threshold=0.3,
    )

    print("Inference pipeline completed successfully.")
    print("\nInference metadata:")
    print(inference_results["inference_metadata"])
    print("\nModel context:")
    print(inference_results["model_context"])
    print("\nModel summary:")
    print(inference_results["model_summary"])
    print("\nPrediction results:")
    print(inference_results["prediction_results"])