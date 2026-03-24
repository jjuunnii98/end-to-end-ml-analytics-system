from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SENIOR_CITIZEN_COLUMN = "SeniorCitizen"


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


def build_numeric_transformer() -> Pipeline:
    """
    Build the numeric preprocessing pipeline.

    This matches the logic from `02_feature_engineering.ipynb`:
    - median imputation
    - standard scaling

    Returns
    -------
    Pipeline
        Numeric preprocessing pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return numeric_transformer


def build_categorical_transformer() -> Pipeline:
    """
    Build the categorical preprocessing pipeline.

    This matches the logic from `02_feature_engineering.ipynb`:
    - most frequent imputation
    - one-hot encoding with unknown-category safety

    Returns
    -------
    Pipeline
        Categorical preprocessing pipeline.
    """
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return categorical_transformer


def build_preprocessor(
    X: pd.DataFrame,
    senior_citizen_col: str = SENIOR_CITIZEN_COLUMN,
) -> ColumnTransformer:
    """
    Build the full ColumnTransformer preprocessor.

    Parameters
    ----------
    X : pd.DataFrame
        Modeling feature dataframe.
    senior_citizen_col : str, default="SeniorCitizen"
        Column that should be treated as categorical.

    Returns
    -------
    ColumnTransformer
        Combined preprocessing transformer for numeric and categorical features.
    """
    numeric_features, categorical_features = get_feature_types(
        X,
        senior_citizen_col=senior_citizen_col,
    )

    numeric_transformer = build_numeric_transformer()
    categorical_transformer = build_categorical_transformer()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def fit_transform_features(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
):
    """
    Fit the preprocessor on training data and transform both train and validation sets.

    This follows the exact rule used in the notebooks:
    - fit on training data only
    - transform validation data using the fitted preprocessor
    - convert sparse output to dense when necessary

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Preprocessing transformer.
    X_train : pd.DataFrame
        Training features.
    X_valid : pd.DataFrame
        Validation features.

    Returns
    -------
    tuple
        X_train_processed, X_valid_processed
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_valid_processed, "toarray"):
        X_valid_processed = X_valid_processed.toarray()

    return X_train_processed, X_valid_processed


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract transformed feature names from the fitted preprocessor.

    This matches the notebook logic using:
    `preprocessor.get_feature_names_out()`

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing transformer.

    Returns
    -------
    list[str]
        List of transformed feature names.
    """
    return list(preprocessor.get_feature_names_out())


def to_processed_dataframe(
    processed_array,
    feature_names: list[str],
    index: pd.Index,
) -> pd.DataFrame:
    """
    Convert a processed feature array into a pandas DataFrame.

    Parameters
    ----------
    processed_array : array-like
        Processed feature matrix.
    feature_names : list[str]
        Transformed feature names.
    index : pd.Index
        Original dataframe index.

    Returns
    -------
    pd.DataFrame
        Processed feature dataframe.
    """
    return pd.DataFrame(processed_array, columns=feature_names, index=index)


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset
    from src.data.preprocess import split_features_and_target, split_train_valid

    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )
    feature_names = get_transformed_feature_names(preprocessor)
    X_train_df = to_processed_dataframe(X_train_processed, feature_names, X_train.index)
    X_valid_df = to_processed_dataframe(X_valid_processed, feature_names, X_valid.index)

    print("Feature building completed successfully.")
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed valid shape: {X_valid_processed.shape}")
    print(f"Number of transformed features: {len(feature_names)}")
    print(f"Nulls in processed train data: {X_train_df.isnull().sum().sum()}")
    print(f"Nulls in processed valid data: {X_valid_df.isnull().sum().sum()}")
