from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_schema import build_feature_schema_dict, split_feature_types


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


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build the full ColumnTransformer preprocessor.

    Parameters
    ----------
    X : pd.DataFrame
        Modeling feature dataframe.

    Returns
    -------
    ColumnTransformer
        Combined preprocessing transformer for numeric and categorical features.
    """
    numeric_features, categorical_features = split_feature_types(X)

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


def summarize_built_features(
    preprocessor: ColumnTransformer,
    X_train_processed,
    X_valid_processed,
) -> dict:
    """
    Build a lightweight summary of the transformed feature outputs.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing transformer.
    X_train_processed : array-like
        Processed training feature matrix.
    X_valid_processed : array-like
        Processed validation feature matrix.

    Returns
    -------
    dict
        Summary of transformed feature shapes and feature count.
    """
    feature_names = get_transformed_feature_names(preprocessor)

    return {
        "n_transformed_features": len(feature_names),
        "train_shape": tuple(X_train_processed.shape),
        "valid_shape": tuple(X_valid_processed.shape),
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    from src.data.load_data import load_telco_dataset
    from src.data.preprocess import split_features_and_target, split_train_valid

    df = load_telco_dataset()
    X, y = split_features_and_target(df)
    X_train, X_valid, y_train, y_valid = split_train_valid(X, y)
    feature_schema = build_feature_schema_dict(X)

    preprocessor = build_preprocessor(X)
    X_train_processed, X_valid_processed = fit_transform_features(
        preprocessor,
        X_train,
        X_valid,
    )
    feature_names = get_transformed_feature_names(preprocessor)
    feature_summary = summarize_built_features(
        preprocessor,
        X_train_processed,
        X_valid_processed,
    )
    X_train_df = to_processed_dataframe(X_train_processed, feature_names, X_train.index)
    X_valid_df = to_processed_dataframe(X_valid_processed, feature_names, X_valid.index)

    print("Feature building completed successfully.")
    print(f"Processed train shape: {feature_summary['train_shape']}")
    print(f"Processed valid shape: {feature_summary['valid_shape']}")
    print(f"Number of transformed features: {feature_summary['n_transformed_features']}")
    print(f"Nulls in processed train data: {X_train_df.isnull().sum().sum()}")
    print(f"Nulls in processed valid data: {X_valid_df.isnull().sum().sum()}")
    print("Feature schema keys:", list(feature_schema.keys()))
