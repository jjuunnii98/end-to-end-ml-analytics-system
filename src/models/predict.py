from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_schema import get_model_input_columns, validate_model_input_columns


DEFAULT_THRESHOLD = 0.5


def _validate_threshold(threshold: float) -> None:
    """
    Validate the decision threshold used for binary classification.

    Parameters
    ----------
    threshold : float
        Decision threshold.

    Raises
    ------
    ValueError
        If the threshold is outside the valid [0, 1] range.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be between 0 and 1. Received: {threshold}")


def prepare_inference_input(input_data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw inference input into a DataFrame aligned with the project schema.

    Parameters
    ----------
    input_data : dict[str, Any] | pd.DataFrame
        Raw single-record input or input dataframe.

    Returns
    -------
    pd.DataFrame
        Input dataframe ordered by the project model input columns.

    Raises
    ------
    TypeError
        If the input type is not supported.
    ValueError
        If required columns are missing.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError(
            "`input_data` must be either a dictionary or a pandas DataFrame."
        )

    validate_model_input_columns(input_df)

    ordered_columns = get_model_input_columns()
    input_df = input_df[ordered_columns].copy()

    return input_df


def transform_inference_input(preprocessor, input_df: pd.DataFrame):
    """
    Transform inference input using a fitted preprocessor.

    Parameters
    ----------
    preprocessor : object
        Fitted preprocessing transformer.
    input_df : pd.DataFrame
        Raw inference dataframe aligned to the project schema.

    Returns
    -------
    array-like
        Transformed feature matrix ready for prediction.
    """
    transformed = preprocessor.transform(input_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return transformed


def predict_labels(model, transformed_input) -> np.ndarray:
    """
    Generate model-default binary class predictions from a fitted model.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator.
    transformed_input : array-like
        Transformed feature matrix.

    Returns
    -------
    np.ndarray
        Predicted binary labels.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("Model must implement `predict`.")

    return model.predict(transformed_input)


def predict_probabilities(model, transformed_input) -> np.ndarray:
    """
    Generate positive-class probabilities from a fitted model.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator.
    transformed_input : array-like
        Transformed feature matrix.

    Returns
    -------
    np.ndarray
        Positive-class probabilities.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement `predict_proba`.")

    return model.predict_proba(transformed_input)[:, 1]


def apply_prediction_threshold(
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    Convert probability predictions into binary labels using a threshold.

    Parameters
    ----------
    probabilities : np.ndarray
        Positive-class probabilities.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    np.ndarray
        Binary predictions.
    """
    _validate_threshold(threshold)
    return (probabilities >= threshold).astype(int)


def summarize_predictions(
    thresholded_labels: np.ndarray,
    probabilities: np.ndarray,
    model_labels: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """
    Build a lightweight prediction summary for each input record.

    Parameters
    ----------
    thresholded_labels : np.ndarray
        Threshold-based predicted labels.
    probabilities : np.ndarray
        Positive-class probabilities.
    model_labels : np.ndarray | None, default=None
        Model-default predicted labels from `model.predict`.

    Returns
    -------
    list[dict[str, float | int]]
        Prediction summaries containing thresholded labels, probabilities,
        and optionally model-default labels.
    """
    results: list[dict[str, float | int]] = []

    if model_labels is None:
        model_labels = thresholded_labels

    for thresholded_label, probability, model_label in zip(
        thresholded_labels,
        probabilities,
        model_labels,
    ):
        results.append(
            {
                "predicted_label": int(thresholded_label),
                "model_label": int(model_label),
                "churn_probability": float(probability),
            }
        )

    return results


def run_inference(
    model,
    preprocessor,
    input_data: dict[str, Any] | pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """
    Run the full prediction workflow for one or more inference records.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator.
    preprocessor : object
        Fitted preprocessing transformer.
    input_data : dict[str, Any] | pd.DataFrame
        Raw single-record input or input dataframe.
    threshold : float, default=0.5
        Decision threshold for binary classification.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the raw input shape, transformed shape,
        probabilities, model-default labels, thresholded predictions,
        and summarized outputs.
    """
    _validate_threshold(threshold)

    input_df = prepare_inference_input(input_data)
    transformed_input = transform_inference_input(preprocessor, input_df)

    model_labels = predict_labels(model, transformed_input)
    probabilities = predict_probabilities(model, transformed_input)
    thresholded_labels = apply_prediction_threshold(probabilities, threshold=threshold)
    summary = summarize_predictions(
        thresholded_labels=thresholded_labels,
        probabilities=probabilities,
        model_labels=model_labels,
    )

    return {
        "input_shape": tuple(input_df.shape),
        "input_columns": input_df.columns.tolist(),
        "transformed_shape": tuple(transformed_input.shape),
        "threshold": threshold,
        "model_labels": [int(label) for label in model_labels],
        "probabilities": [float(probability) for probability in probabilities],
        "predictions": summary,
    }


if __name__ == "__main__":
    from src.pipelines.training_pipeline import run_training_pipeline

    pipeline_results = run_training_pipeline(
        model_name="random_forest",
        custom_threshold=0.3,
    )
    model = pipeline_results["model"]
    preprocessor = pipeline_results["preprocessor"]

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

    prediction_results = run_inference(
        model=model,
        preprocessor=preprocessor,
        input_data=sample_input,
        threshold=0.3,
    )

    print("Prediction pipeline completed successfully.")
    print(prediction_results)