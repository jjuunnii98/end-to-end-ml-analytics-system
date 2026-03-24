from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.feature_schema import (
    ID_COLUMN,
    ORIGINAL_TARGET_COLUMN,
    TARGET_COLUMN,
    build_feature_schema_dict,
    split_feature_types,
)


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    original_target_col: str = ORIGINAL_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the input dataframe into modeling features (X) and target (y).

    This function follows the exact feature/target separation rule used in
    `02_feature_engineering.ipynb`, `03_model_experiments.ipynb`, and
    the shared project feature schema:
    - exclude `customerID`
    - exclude original string target `Churn`
    - use `Churn_binary` as the modeling target

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the cleaned dataset.
    target_col : str, default="Churn_binary"
        Binary target column used for modeling.
    id_col : str, default="customerID"
        Identifier column to exclude from modeling.
    original_target_col : str, default="Churn"
        Original string target column to exclude from modeling.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X : modeling features
        y : binary target

    Raises
    ------
    KeyError
        If the required target column does not exist.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column `{target_col}` not found in dataframe.")

    exclude_cols = [id_col, original_target_col, target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def split_train_valid(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and validation sets.

    This function follows the exact split rule used in the notebooks:
    - validation size = 20%
    - random_state = 42
    - stratify by target to preserve churn ratio

    Parameters
    ----------
    X : pd.DataFrame
        Modeling features.
    y : pd.Series
        Binary target.
    test_size : float, default=0.2
        Validation set proportion.
    random_state : int, default=42
        Random seed for reproducibility.
    stratify : bool, default=True
        Whether to apply stratified split.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_valid, y_train, y_valid
    """
    stratify_target = y if stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    return X_train, X_valid, y_train, y_valid


def summarize_preprocessing_inputs(X: pd.DataFrame) -> dict:
    """
    Build a lightweight preprocessing summary based on the shared feature schema.

    Parameters
    ----------
    X : pd.DataFrame
        Modeling feature dataframe.

    Returns
    -------
    dict
        Dictionary containing numeric and categorical feature counts.
    """
    numeric_features, categorical_features = split_feature_types(X)

    return {
        "n_features": int(X.shape[1]),
        "n_numeric_features": len(numeric_features),
        "n_categorical_features": len(categorical_features),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset

    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    numeric_features, categorical_features = split_feature_types(X)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)
    feature_schema = build_feature_schema_dict(X)
    preprocessing_summary = summarize_preprocessing_inputs(X)

    print("Preprocessing split completed successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"Number of numeric features: {preprocessing_summary['n_numeric_features']}")
    print(f"Number of categorical features: {preprocessing_summary['n_categorical_features']}")
    print("Feature schema keys:", list(feature_schema.keys()))
    print("Excluded columns:", feature_schema["excluded_columns"])
