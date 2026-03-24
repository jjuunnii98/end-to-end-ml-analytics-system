from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from src.data.load_data import load_telco_dataset, summarize_dataset
from src.data.validate_data import validate_dataset
from src.data.preprocess import (
    split_features_and_target,
    split_train_valid,
    summarize_preprocessing_inputs,
)
from src.features.feature_schema import build_feature_schema_dict
from src.features.build_features import (
    build_preprocessor,
    fit_transform_features,
    summarize_built_features,
)
from src.models.train import SUPPORTED_MODELS, train_named_model, summarize_trained_model
from src.models.evaluate import summarize_evaluation


DEFAULT_MODEL_NAME = "random_forest"
DEFAULT_THRESHOLD = 0.3
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_MODEL_DIR = DEFAULT_ARTIFACTS_DIR / "model"
DEFAULT_METRICS_DIR = DEFAULT_ARTIFACTS_DIR / "metrics"


def _validate_pipeline_inputs(
    model_name: str,
    custom_threshold: float,
) -> None:
    """
    Validate user-provided training pipeline inputs.

    Parameters
    ----------
    model_name : str
        Model name requested for training.
    custom_threshold : float
        Threshold used for threshold-level evaluation.

    Raises
    ------
    ValueError
        If the model name is unsupported or the threshold is outside [0, 1].
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models: {sorted(SUPPORTED_MODELS)}"
        )

    if not 0.0 <= custom_threshold <= 1.0:
        raise ValueError(
            f"`custom_threshold` must be between 0 and 1. Received: {custom_threshold}"
        )


def _build_pipeline_metadata(
    model_name: str,
    custom_threshold: float,
    data_path: str | Path | None,
) -> dict[str, Any]:
    """
    Build lightweight metadata for the current training pipeline run.

    Parameters
    ----------
    model_name : str
        Selected training model.
    custom_threshold : float
        Threshold used for threshold-level evaluation.
    data_path : str | Path | None
        Optional custom data path.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary describing the current run.
    """
    resolved_data_path = str(data_path) if data_path is not None else "default_project_path"

    return {
        "model_name": model_name,
        "custom_threshold": custom_threshold,
        "data_path": resolved_data_path,
    }


def _ensure_artifact_directories(
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> None:
    """
    Ensure artifact output directories exist.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


def _save_json_artifact(payload: dict[str, Any], output_path: Path) -> Path:
    """
    Save a JSON artifact to disk.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def _normalize_for_json(value: Any) -> Any:
    """
    Convert numpy/scalar-like values into JSON-serializable Python values.
    """
    if isinstance(value, dict):
        return {k: _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_for_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def save_training_artifacts(
    model,
    preprocessor,
    evaluation_summary: dict[str, Any],
    model_name: str,
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
) -> dict[str, str]:
    """
    Save model, preprocessor, and evaluation summary artifacts.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible model.
    preprocessor : object
        Fitted preprocessing transformer.
    evaluation_summary : dict[str, Any]
        Evaluation summary dictionary.
    model_name : str
        Model name used for file naming.
    model_dir : Path, default=artifacts/model
        Directory for serialized model artifacts.
    metrics_dir : Path, default=artifacts/metrics
        Directory for evaluation artifacts.

    Returns
    -------
    dict[str, str]
        Paths to the saved artifacts.
    """
    _ensure_artifact_directories(model_dir=model_dir, metrics_dir=metrics_dir)

    model_path = model_dir / f"{model_name}_model.joblib"
    preprocessor_path = model_dir / f"{model_name}_preprocessor.joblib"
    metrics_path = metrics_dir / f"{model_name}_evaluation.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    _save_json_artifact(_normalize_for_json(evaluation_summary), metrics_path)

    return {
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "metrics_path": str(metrics_path),
    }


def run_training_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    custom_threshold: float = DEFAULT_THRESHOLD,
    data_path: str | Path | None = None,
    save_artifacts: bool = True,
    model_dir: Path = DEFAULT_MODEL_DIR,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """
    Run the full training pipeline for the Telco churn prediction project.

    Pipeline steps
    --------------
    1. Load cleaned dataset
    2. Validate dataset structure and values
    3. Split features and target
    4. Create train/validation split
    5. Build feature preprocessor
    6. Transform train/validation features
    7. Train selected model
    8. Evaluate trained model
    9. Save model/preprocessor/metrics artifacts
    10. Return pipeline summary artifacts

    Parameters
    ----------
    model_name : str, default="random_forest"
        Model name to train.
    custom_threshold : float, default=0.3
        Threshold used for threshold-level evaluation.
    data_path : str | Path | None, default=None
        Optional custom path to the Telco CSV file.
    save_artifacts : bool, default=True
        Whether to save the trained model, preprocessor, and evaluation summary.
    model_dir : Path, default=artifacts/model
        Directory used to save serialized model artifacts.
    metrics_dir : Path, default=artifacts/metrics
        Directory used to save evaluation artifacts.
    **model_kwargs : Any
        Additional keyword arguments passed to the model builder.

    Returns
    -------
    dict[str, Any]
        Dictionary containing pipeline objects and summary outputs.
    """
    _validate_pipeline_inputs(model_name=model_name, custom_threshold=custom_threshold)
    pipeline_metadata = _build_pipeline_metadata(
        model_name=model_name,
        custom_threshold=custom_threshold,
        data_path=data_path,
    )

    # 1. Load dataset using the shared project data-loading rules
    df = load_telco_dataset(data_path=data_path)
    dataset_summary = summarize_dataset(df)

    # 2. Validate dataset before downstream processing
    validate_dataset(df)

    # 3. Split features and target using the shared schema rules
    X, y = split_features_and_target(df)
    feature_schema = build_feature_schema_dict(X)
    preprocessing_summary = summarize_preprocessing_inputs(X)

    # 4. Create train/validation split
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    # 5. Build preprocessor from feature schema rules
    preprocessor = build_preprocessor(X)

    # 6. Transform features for model training and validation
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )
    built_feature_summary = summarize_built_features(
        preprocessor,
        X_train_processed,
        X_valid_processed,
    )

    # 7. Train selected model
    model = train_named_model(
        model_name=model_name,
        X_train=X_train_processed,
        y_train=y_train,
        **model_kwargs,
    )
    model_summary = summarize_trained_model(model)

    # 8. Evaluate trained model
    evaluation_summary = summarize_evaluation(
        model=model,
        X_valid=X_valid_processed,
        y_valid=y_valid,
        model_name=model_name,
        custom_threshold=custom_threshold,
    )

    artifact_paths: dict[str, str] = {}
    if save_artifacts:
        artifact_paths = save_training_artifacts(
            model=model,
            preprocessor=preprocessor,
            evaluation_summary=evaluation_summary,
            model_name=model_name,
            model_dir=model_dir,
            metrics_dir=metrics_dir,
        )

    # 10. Return all core artifacts for downstream saving / deployment
    return {
        "pipeline_metadata": pipeline_metadata,
        "dataset_summary": dataset_summary,
        "feature_schema": feature_schema,
        "preprocessing_summary": preprocessing_summary,
        "built_feature_summary": built_feature_summary,
        "model_summary": model_summary,
        "evaluation_summary": evaluation_summary,
        "artifact_paths": artifact_paths,
        "dataframe": df,
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "X_train_processed": X_train_processed,
        "X_valid_processed": X_valid_processed,
        "preprocessor": preprocessor,
        "model": model,
    }


if __name__ == "__main__":
    pipeline_results = run_training_pipeline()

    print("Training pipeline completed successfully.")

    print("\nPipeline metadata:")
    print(pipeline_results["pipeline_metadata"])

    print("\nDataset summary:")
    print(pipeline_results["dataset_summary"])

    print("\nFeature schema keys:")
    print(list(pipeline_results["feature_schema"].keys()))

    print("\nPreprocessing summary:")
    print(pipeline_results["preprocessing_summary"])

    print("\nBuilt feature summary:")
    print(pipeline_results["built_feature_summary"])

    print("\nModel summary:")
    print(pipeline_results["model_summary"])

    print("\nEvaluation summary:")
    print(pipeline_results["evaluation_summary"])

    print("\nArtifact paths:")
    print(pipeline_results["artifact_paths"])