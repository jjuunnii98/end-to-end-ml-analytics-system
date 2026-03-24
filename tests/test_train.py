from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.load_data import load_telco_dataset
from src.data.preprocess import split_features_and_target, split_train_valid
from src.features.build_features import build_preprocessor, fit_transform_features
from src.models.train import (
    SUPPORTED_MODELS,
    build_logistic_regression_model,
    build_model,
    build_random_forest_model,
    summarize_trained_model,
    train_model,
    train_named_model,
)


def _get_processed_training_data():
    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )

    return X_train_processed, X_valid_processed, y_train, y_valid



def test_supported_models_contains_expected_model_names() -> None:
    assert SUPPORTED_MODELS == {"logistic_regression", "random_forest"}



def test_build_logistic_regression_model_returns_expected_estimator() -> None:
    model = build_logistic_regression_model()

    assert isinstance(model, LogisticRegression)
    assert model.max_iter == 1000
    assert model.random_state == 42



def test_build_random_forest_model_returns_expected_estimator() -> None:
    model = build_random_forest_model()

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.random_state == 42



def test_build_model_returns_logistic_regression_for_supported_name() -> None:
    model = build_model("logistic_regression")

    assert isinstance(model, LogisticRegression)



def test_build_model_returns_random_forest_for_supported_name() -> None:
    model = build_model("random_forest")

    assert isinstance(model, RandomForestClassifier)



def test_build_model_raises_for_unsupported_model_name() -> None:
    try:
        build_model("xgboost")
        assert False, "Expected ValueError for unsupported model name"
    except ValueError as exc:
        assert "Unsupported model" in str(exc)



def test_train_model_fits_logistic_regression_successfully() -> None:
    X_train_processed, _, y_train, _ = _get_processed_training_data()
    model = build_logistic_regression_model()

    fitted_model = train_model(model, X_train_processed, y_train)

    assert hasattr(fitted_model, "coef_")
    assert fitted_model.n_features_in_ == 46



def test_train_named_model_fits_random_forest_successfully() -> None:
    X_train_processed, _, y_train, _ = _get_processed_training_data()

    fitted_model = train_named_model(
        model_name="random_forest",
        X_train=X_train_processed,
        y_train=y_train,
    )

    assert isinstance(fitted_model, RandomForestClassifier)
    assert hasattr(fitted_model, "feature_importances_")
    assert fitted_model.n_features_in_ == 46



def test_summarize_trained_model_returns_expected_keys_for_random_forest() -> None:
    X_train_processed, _, y_train, _ = _get_processed_training_data()
    fitted_model = train_named_model(
        model_name="random_forest",
        X_train=X_train_processed,
        y_train=y_train,
    )

    summary = summarize_trained_model(fitted_model)

    assert summary["model_class"] == "RandomForestClassifier"
    assert summary["has_predict"] is True
    assert summary["has_predict_proba"] is True
    assert summary["n_features_in"] == 46
    assert summary["classes"] == [0, 1]



def test_summarize_trained_model_returns_expected_keys_for_logistic_regression() -> None:
    X_train_processed, _, y_train, _ = _get_processed_training_data()
    fitted_model = train_named_model(
        model_name="logistic_regression",
        X_train=X_train_processed,
        y_train=y_train,
    )

    summary = summarize_trained_model(fitted_model)

    assert summary["model_class"] == "LogisticRegression"
    assert summary["has_predict"] is True
    assert summary["has_predict_proba"] is True
    assert summary["n_features_in"] == 46
    assert summary["classes"] == [0, 1]
