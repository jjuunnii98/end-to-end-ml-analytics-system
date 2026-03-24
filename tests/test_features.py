

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.load_data import load_telco_dataset
from src.data.preprocess import split_features_and_target, split_train_valid
from src.features.build_features import (
    build_categorical_transformer,
    build_numeric_transformer,
    build_preprocessor,
    fit_transform_features,
    get_transformed_feature_names,
    summarize_built_features,
    to_processed_dataframe,
)


def test_build_numeric_transformer_returns_pipeline() -> None:
    numeric_transformer = build_numeric_transformer()

    assert isinstance(numeric_transformer, Pipeline)
    assert list(numeric_transformer.named_steps.keys()) == ["imputer", "scaler"]


def test_build_categorical_transformer_returns_pipeline() -> None:
    categorical_transformer = build_categorical_transformer()

    assert isinstance(categorical_transformer, Pipeline)
    assert list(categorical_transformer.named_steps.keys()) == ["imputer", "onehot"]


def test_build_preprocessor_returns_column_transformer() -> None:
    df = load_telco_dataset()
    X, _ = split_features_and_target(df)

    preprocessor = build_preprocessor(X)

    assert isinstance(preprocessor, ColumnTransformer)
    assert [name for name, _, _ in preprocessor.transformers] == ["num", "cat"]


def test_fit_transform_features_returns_expected_shapes() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, _, _ = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )

    assert X_train_processed.shape == (5625, 46)
    assert X_valid_processed.shape == (1407, 46)


def test_get_transformed_feature_names_returns_expected_feature_count() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, _, _ = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    fit_transform_features(preprocessor, X_train, X_valid)
    feature_names = get_transformed_feature_names(preprocessor)

    assert len(feature_names) == 46
    assert "num__MonthlyCharges" in feature_names
    assert "num__TotalCharges" in feature_names
    assert "num__tenure" in feature_names
    assert "cat__SeniorCitizen_0" in feature_names
    assert "cat__SeniorCitizen_1" in feature_names


def test_to_processed_dataframe_returns_dataframe_with_expected_columns() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, _, _ = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )
    feature_names = get_transformed_feature_names(preprocessor)

    X_train_df = to_processed_dataframe(X_train_processed, feature_names, X_train.index)
    X_valid_df = to_processed_dataframe(X_valid_processed, feature_names, X_valid.index)

    assert X_train_df.shape == (5625, 46)
    assert X_valid_df.shape == (1407, 46)
    assert list(X_train_df.columns) == feature_names
    assert X_train_df.isnull().sum().sum() == 0
    assert X_valid_df.isnull().sum().sum() == 0


def test_summarize_built_features_returns_expected_summary() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, _, _ = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )

    summary = summarize_built_features(
        preprocessor,
        X_train_processed,
        X_valid_processed,
    )

    assert summary["n_transformed_features"] == 46
    assert summary["train_shape"] == (5625, 46)
    assert summary["valid_shape"] == (1407, 46)
    assert len(summary["feature_names"]) == 46