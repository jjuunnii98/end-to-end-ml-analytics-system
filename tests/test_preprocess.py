

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import load_telco_dataset
from src.data.preprocess import (
    split_features_and_target,
    split_train_valid,
    summarize_preprocessing_inputs,
)
from src.features.feature_schema import (
    ORIGINAL_TARGET_COLUMN,
    TARGET_COLUMN,
    ID_COLUMN,
)


def test_split_features_and_target_excludes_id_and_targets() -> None:
    df = load_telco_dataset()

    X, y = split_features_and_target(df)

    assert ID_COLUMN not in X.columns
    assert ORIGINAL_TARGET_COLUMN not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert y.name == TARGET_COLUMN
    assert len(X) == len(y)



def test_split_features_and_target_returns_expected_feature_count() -> None:
    df = load_telco_dataset()

    X, y = split_features_and_target(df)

    assert X.shape[1] == 19
    assert y.nunique() == 2
    assert set(y.unique()) == {0, 1}



def test_split_train_valid_preserves_total_row_count() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)

    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    assert len(X_train) + len(X_valid) == len(X)
    assert len(y_train) + len(y_valid) == len(y)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_valid.shape[0] == y_valid.shape[0]



def test_split_train_valid_matches_expected_shapes() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)

    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    assert X_train.shape == (5625, 19)
    assert X_valid.shape == (1407, 19)
    assert y_train.shape[0] == 5625
    assert y_valid.shape[0] == 1407



def test_split_train_valid_preserves_class_ratio_reasonably() -> None:
    df = load_telco_dataset()
    X, y = split_features_and_target(df)

    _, _, y_train, y_valid = split_train_valid(X, y)

    train_ratio = y_train.mean()
    valid_ratio = y_valid.mean()

    assert abs(train_ratio - valid_ratio) < 0.01



def test_summarize_preprocessing_inputs_returns_expected_counts() -> None:
    df = load_telco_dataset()
    X, _ = split_features_and_target(df)

    summary = summarize_preprocessing_inputs(X)

    assert summary["n_features"] == 19
    assert summary["n_numeric_features"] == 3
    assert summary["n_categorical_features"] == 16



def test_summarize_preprocessing_inputs_contains_expected_feature_groups() -> None:
    df = load_telco_dataset()
    X, _ = split_features_and_target(df)

    summary = summarize_preprocessing_inputs(X)

    assert summary["numeric_features"] == ["MonthlyCharges", "TotalCharges", "tenure"]
    assert "SeniorCitizen" in summary["categorical_features"]
    assert "Contract" in summary["categorical_features"]
    assert "PaymentMethod" in summary["categorical_features"]



def test_split_features_and_target_raises_for_missing_target() -> None:
    df = load_telco_dataset().drop(columns=[TARGET_COLUMN])

    try:
        split_features_and_target(df)
        assert False, "Expected KeyError for missing target column"
    except KeyError as exc:
        assert TARGET_COLUMN in str(exc)