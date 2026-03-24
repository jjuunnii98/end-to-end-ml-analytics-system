

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLD = 0.5


def generate_predictions(model, X_valid):
    """
    Generate class predictions and probability predictions from a fitted model.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator.
    X_valid : array-like
        Processed validation feature matrix.

    Returns
    -------
    tuple
        pred_labels, pred_proba

    Raises
    ------
    AttributeError
        If the model does not implement the required prediction methods.
    """
    if not hasattr(model, "predict"):
        raise AttributeError("Model must implement `predict`.")

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model must implement `predict_proba`.")

    pred_labels = model.predict(X_valid)
    pred_proba = model.predict_proba(X_valid)[:, 1]

    return pred_labels, pred_proba


def evaluate_classification_metrics(
    y_true,
    y_pred,
    y_proba,
    model_name: str,
) -> dict[str, Any]:
    """
    Evaluate standard binary classification metrics.

    This matches the evaluation logic used in `03_model_experiments.ipynb`.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary class labels.
    y_proba : array-like
        Predicted positive-class probabilities.
    model_name : str
        Model name for reporting.

    Returns
    -------
    dict[str, Any]
        Dictionary containing classification metrics.
    """
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def apply_threshold(y_proba, threshold: float = DEFAULT_THRESHOLD):
    """
    Convert probability predictions into binary class predictions using a threshold.

    Parameters
    ----------
    y_proba : array-like
        Predicted positive-class probabilities.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    array-like
        Thresholded binary predictions.
    """
    return (y_proba >= threshold).astype(int)


def evaluate_threshold_metrics(
    y_true,
    y_proba,
    threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Evaluate precision and recall under a custom decision threshold.

    This mirrors the threshold analysis performed in
    `03_model_experiments.ipynb`.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_proba : array-like
        Predicted positive-class probabilities.
    threshold : float, default=0.3
        Custom decision threshold.

    Returns
    -------
    dict[str, Any]
        Dictionary containing threshold-level precision and recall.
    """
    y_pred_custom = apply_threshold(y_proba, threshold=threshold)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred_custom),
        "recall": recall_score(y_true, y_pred_custom),
    }


def compare_model_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of metric dictionaries into a comparison DataFrame.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of model metric dictionaries.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    return pd.DataFrame(results)


def rank_model_results(
    results_df: pd.DataFrame,
    metric: str = "roc_auc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Rank model results by a primary metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Model comparison dataframe.
    metric : str, default="roc_auc"
        Metric column used for ranking.
    ascending : bool, default=False
        Sort order.

    Returns
    -------
    pd.DataFrame
        Sorted model comparison dataframe.
    """
    return results_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)


def summarize_evaluation(
    model,
    X_valid,
    y_valid,
    model_name: str,
    custom_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Run a full evaluation summary for a fitted model.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible estimator.
    X_valid : array-like
        Processed validation feature matrix.
    y_valid : array-like
        Validation target.
    model_name : str
        Model name for reporting.
    custom_threshold : float, default=0.3
        Additional threshold used for threshold analysis.

    Returns
    -------
    dict[str, Any]
        Combined evaluation summary.
    """
    y_pred, y_proba = generate_predictions(model, X_valid)
    default_metrics = evaluate_classification_metrics(
        y_true=y_valid,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name=model_name,
    )
    threshold_metrics = evaluate_threshold_metrics(
        y_true=y_valid,
        y_proba=y_proba,
        threshold=custom_threshold,
    )

    return {
        "default_metrics": default_metrics,
        "threshold_metrics": threshold_metrics,
    }


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset
    from src.data.preprocess import split_features_and_target, split_train_valid
    from src.features.build_features import build_preprocessor, fit_transform_features
    from src.models.train import train_named_model

    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )

    logistic_model = train_named_model(
        model_name="logistic_regression",
        X_train=X_train_processed,
        y_train=y_train,
    )
    random_forest_model = train_named_model(
        model_name="random_forest",
        X_train=X_train_processed,
        y_train=y_train,
    )

    logistic_eval = summarize_evaluation(
        model=logistic_model,
        X_valid=X_valid_processed,
        y_valid=y_valid,
        model_name="Logistic Regression",
    )
    random_forest_eval = summarize_evaluation(
        model=random_forest_model,
        X_valid=X_valid_processed,
        y_valid=y_valid,
        model_name="Random Forest",
    )

    results_df = compare_model_results([
        logistic_eval["default_metrics"],
        random_forest_eval["default_metrics"],
    ])
    ranked_results_df = rank_model_results(results_df, metric="roc_auc", ascending=False)

    print("Model evaluation completed successfully.")
    print("\nLogistic Regression evaluation:")
    print(logistic_eval)
    print("\nRandom Forest evaluation:")
    print(random_forest_eval)
    print("\nRanked results:")
    print(ranked_results_df)