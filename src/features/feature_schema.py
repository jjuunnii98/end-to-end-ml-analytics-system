from __future__ import annotations

from typing import Tuple

import pandas as pd


ID_COLUMN = "customerID"
ORIGINAL_TARGET_COLUMN = "Churn"
TARGET_COLUMN = "Churn_binary"
SENIOR_CITIZEN_COLUMN = "SeniorCitizen"

EXCLUDED_COLUMNS = [
    ID_COLUMN,
    ORIGINAL_TARGET_COLUMN,
    TARGET_COLUMN,
]

RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

FORCED_CATEGORICAL_COLUMNS = [
    SENIOR_CITIZEN_COLUMN,
]


def get_model_input_columns() -> list[str]:
    """
    Return the ordered list of raw input columns used for modeling.
    """
    return RAW_FEATURE_COLUMNS.copy()


def get_excluded_columns() -> list[str]:
    """
    Return columns excluded from modeling.
    """
    return EXCLUDED_COLUMNS.copy()


def split_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """
    Split features into numeric and categorical groups using project rules.

    Rules
    -----
    - numeric: int64/float64 columns
    - categorical: object columns
    - SeniorCitizen is always treated as categorical
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    for col in FORCED_CATEGORICAL_COLUMNS:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in X.columns and col not in categorical_features:
            categorical_features.append(col)

    numeric_features = sorted(numeric_features)
    categorical_features = sorted(categorical_features)

    return numeric_features, categorical_features


def validate_model_input_columns(df: pd.DataFrame) -> None:
    """
    Validate whether the dataframe contains all required raw feature columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = set(get_model_input_columns())
    observed_cols = set(df.columns)

    missing_cols = sorted(required_cols - observed_cols)
    if missing_cols:
        raise ValueError(f"Missing required model input columns: {missing_cols}")


def build_feature_schema_dict(X: pd.DataFrame) -> dict:
    """
    Build a lightweight schema dictionary for debugging, logging, and pipeline usage.
    """
    numeric_features, categorical_features = split_feature_types(X)

    return {
        "id_column": ID_COLUMN,
        "original_target_column": ORIGINAL_TARGET_COLUMN,
        "target_column": TARGET_COLUMN,
        "excluded_columns": get_excluded_columns(),
        "input_columns": get_model_input_columns(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "forced_categorical_columns": FORCED_CATEGORICAL_COLUMNS.copy(),
    }


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset
    from src.data.preprocess import split_features_and_target

    df = load_telco_dataset()
    X, y = split_features_and_target(df)

    schema = build_feature_schema_dict(X)

    print("Feature schema loaded successfully.")
    print("Input columns:", schema["input_columns"])
    print("Numeric features:", schema["numeric_features"])
    print("Categorical features:", schema["categorical_features"])