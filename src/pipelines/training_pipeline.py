from __future__ import annotations

from pathlib import Path
from typing import Any

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


def run_training_pipeline(
    model_name: str = DEFAULT_MODEL_NAME,
    custom_threshold: float = DEFAULT_THRESHOLD,
    data_path: str | Path | None = None,
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
    9. Return pipeline summary artifacts

    Parameters
    ----------
    model_name : str, default="random_forest"
        Model name to train.
    custom_threshold : float, default=0.3
        Threshold used for threshold-level evaluation.
    data_path : str | Path | None, default=None
        Optional custom path to the Telco CSV file.
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

    # 9. Return all core artifacts for downstream saving / deployment
    return {
        "pipeline_metadata": pipeline_metadata,
        "dataset_summary": dataset_summary,
        "feature_schema": feature_schema,
        "preprocessing_summary": preprocessing_summary,
        "built_feature_summary": built_feature_summary,
        "model_summary": model_summary,
        "evaluation_summary": evaluation_summary,
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