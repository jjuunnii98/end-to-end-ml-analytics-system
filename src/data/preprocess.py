from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMN = "customerID"
ORIGINAL_TARGET_COLUMN = "Churn"
TARGET_COLUMN = "Churn_binary"
SENIOR_CITIZEN_COLUMN = "SeniorCitizen"


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    original_target_col: str = ORIGINAL_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the input dataframe into modeling features (X) and target (y).

    This function follows the exact feature/target separation rule used in
    `02_feature_engineering.ipynb` and `03_model_experiments.ipynb`:
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


def get_feature_types(
    X: pd.DataFrame,
    senior_citizen_col: str = SENIOR_CITIZEN_COLUMN,
) -> Tuple[list[str], list[str]]:
    """
    Separate numeric and categorical feature columns.

    This function follows the exact rule used in the notebooks:
    - numeric features are selected from int64/float64 columns
    - categorical features are selected from object columns
    - `SeniorCitizen` is treated as a categorical/binary feature,
      even if it is numerically encoded

    Parameters
    ----------
    X : pd.DataFrame
        Modeling feature dataframe.
    senior_citizen_col : str, default="SeniorCitizen"
        Column that should be treated as categorical.

    Returns
    -------
    Tuple[list[str], list[str]]
        numeric_features, categorical_features
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    if senior_citizen_col in numeric_features:
        numeric_features.remove(senior_citizen_col)

    if senior_citizen_col in X.columns and senior_citizen_col not in categorical_features:
        categorical_features.append(senior_citizen_col)

    numeric_features = sorted(numeric_features)
    categorical_features = sorted(categorical_features)

    return numeric_features, categorical_features


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


def build_feature_schema(
    X: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
) -> dict:
    """
    Build a lightweight feature schema dictionary.

    This mirrors the feature schema preview created in
    `02_feature_engineering.ipynb`.

    Parameters
    ----------
    X : pd.DataFrame
        Modeling feature dataframe.
    target_col : str, default="Churn_binary"
        Target column name.

    Returns
    -------
    dict
        Dictionary containing numeric features, categorical features, and target.
    """
    numeric_features, categorical_features = get_feature_types(X)

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target_col,
    }


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset

    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    numeric_features, categorical_features = get_feature_types(X)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)
    feature_schema = build_feature_schema(X)

    print("Preprocessing split completed successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"Number of numeric features: {len(numeric_features)}")
    print(f"Number of categorical features: {len(categorical_features)}")
    print("Feature schema keys:", list(feature_schema.keys()))
