

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


DEFAULT_RANDOM_STATE = 42
DEFAULT_LOGISTIC_MAX_ITER = 1000
DEFAULT_RF_N_ESTIMATORS = 100
DEFAULT_RF_MAX_DEPTH = 10

SUPPORTED_MODELS = {
    "logistic_regression",
    "random_forest",
}


def build_logistic_regression_model(
    max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LogisticRegression:
    """
    Build the baseline Logistic Regression model.

    This matches the modeling logic used in `03_model_experiments.ipynb`.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Configured Logistic Regression model.
    """
    return LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )


def build_random_forest_model(
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    max_depth: int = DEFAULT_RF_MAX_DEPTH,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Build the Random Forest model.

    This matches the modeling logic used in `03_model_experiments.ipynb`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=10
        Maximum depth of the trees.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Configured Random Forest model.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )


def build_model(model_name: str, **kwargs: Any):
    """
    Build a model by name.

    Supported models
    ----------------
    - logistic_regression
    - random_forest

    Parameters
    ----------
    model_name : str
        Name of the model to build.
    **kwargs : Any
        Additional keyword arguments passed to the model builder.

    Returns
    -------
    object
        Configured sklearn model instance.

    Raises
    ------
    ValueError
        If an unsupported model name is provided.
    """
    if model_name == "logistic_regression":
        return build_logistic_regression_model(**kwargs)

    if model_name == "random_forest":
        return build_random_forest_model(**kwargs)

    raise ValueError(
        f"Unsupported model: {model_name}. Supported models: {sorted(SUPPORTED_MODELS)}"
    )


def train_model(model, X_train, y_train):
    """
    Fit a model on training data.

    Parameters
    ----------
    model : object
        Sklearn-compatible estimator.
    X_train : array-like
        Processed training features.
    y_train : array-like
        Training target.

    Returns
    -------
    object
        Fitted model.
    """
    model.fit(X_train, y_train)
    return model


def train_named_model(
    model_name: str,
    X_train,
    y_train,
    **kwargs: Any,
):
    """
    Build and train a named model in one step.

    Parameters
    ----------
    model_name : str
        Name of the model to build and train.
    X_train : array-like
        Processed training features.
    y_train : array-like
        Training target.
    **kwargs : Any
        Additional model builder keyword arguments.

    Returns
    -------
    object
        Fitted sklearn model.
    """
    model = build_model(model_name, **kwargs)
    fitted_model = train_model(model, X_train, y_train)
    return fitted_model


def summarize_trained_model(model) -> dict[str, Any]:
    """
    Build a lightweight summary of a trained model for logging/debugging.

    Parameters
    ----------
    model : object
        Fitted sklearn model.

    Returns
    -------
    dict[str, Any]
        Summary dictionary containing model class and basic metadata.
    """
    summary = {
        "model_class": model.__class__.__name__,
        "has_predict": hasattr(model, "predict"),
        "has_predict_proba": hasattr(model, "predict_proba"),
    }

    if hasattr(model, "n_features_in_"):
        summary["n_features_in"] = int(model.n_features_in_)

    if hasattr(model, "classes_"):
        summary["classes"] = list(model.classes_)

    return summary


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset
    from src.data.preprocess import split_features_and_target, split_train_valid
    from src.features.build_features import build_preprocessor, fit_transform_features

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

    logistic_summary = summarize_trained_model(logistic_model)
    random_forest_summary = summarize_trained_model(random_forest_model)

    print("Model training completed successfully.")
    print("Logistic Regression summary:", logistic_summary)
    print("Random Forest summary:", random_forest_summary)