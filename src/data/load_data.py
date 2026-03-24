from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_raw_data(data_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset from CSV.

    Parameters
    ----------
    data_path : Optional[str | Path], default=None
        Path to the raw CSV file. If None, the default project data path is used.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded from CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    path = Path(data_path) if data_path is not None else DEFAULT_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert `TotalCharges` to numeric and coerce invalid values to NaN.

    This follows the same rule used in the EDA and feature engineering notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the `TotalCharges` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with `TotalCharges` converted to numeric.

    Raises
    ------
    KeyError
        If `TotalCharges` column does not exist.
    """
    if "TotalCharges" not in df.columns:
        raise KeyError("Column `TotalCharges` not found in dataframe.")

    cleaned_df = df.copy()
    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors="coerce",
    )
    return cleaned_df


def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target column `Churn_binary` from `Churn`.

    Mapping:
    - No  -> 0
    - Yes -> 1

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the `Churn` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the `Churn_binary` column added.

    Raises
    ------
    KeyError
        If `Churn` column does not exist.
    ValueError
        If unexpected target values are found.
    """
    if "Churn" not in df.columns:
        raise KeyError("Column `Churn` not found in dataframe.")

    cleaned_df = df.copy()

    valid_values = {"Yes", "No"}
    observed_values = set(cleaned_df["Churn"].dropna().unique())

    if not observed_values.issubset(valid_values):
        raise ValueError(
            f"Unexpected values found in `Churn`: {sorted(observed_values - valid_values)}"
        )

    cleaned_df["Churn_binary"] = cleaned_df["Churn"].map({"No": 0, "Yes": 1})
    return cleaned_df


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows containing missing values.

    In this project, this is mainly used after `TotalCharges` coercion,
    where invalid string values become NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing rows removed.
    """
    return df.dropna().copy()


def load_telco_dataset(
    data_path: Optional[str | Path] = None,
    clean_totalcharges: bool = True,
    drop_missing: bool = True,
    add_target: bool = True,
) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset with project-consistent cleaning rules.

    This function reproduces the core data-loading logic used in:
    - 01_eda.ipynb
    - 02_feature_engineering.ipynb
    - 03_model_experiments.ipynb

    Parameters
    ----------
    data_path : Optional[str | Path], default=None
        Path to the raw CSV file.
    clean_totalcharges : bool, default=True
        Whether to convert `TotalCharges` to numeric.
    drop_missing : bool, default=True
        Whether to drop rows with missing values after conversion.
    add_target : bool, default=True
        Whether to add the `Churn_binary` target column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for downstream preprocessing and modeling.
    """
    df = load_raw_data(data_path=data_path)

    if clean_totalcharges:
        df = clean_total_charges(df)

    if drop_missing:
        df = drop_missing_rows(df)

    if add_target:
        df = add_binary_target(df)

    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Return a lightweight summary of the dataset for debugging and logging.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Summary dictionary containing shape, missing count, and columns.
    """
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "columns": df.columns.tolist(),
    }


if __name__ == "__main__":
    dataset = load_telco_dataset()
    summary = summarize_dataset(dataset)

    print("Loaded Telco dataset successfully.")
    print(f"Shape: ({summary['n_rows']}, {summary['n_cols']})")
    print(f"Missing values: {summary['missing_values']}")
    print("First 5 columns:", summary["columns"][:5])