import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "Churn",
    "TotalCharges",
    "tenure",
    "MonthlyCharges"
]

EXPECTED_DTYPES = {
    "customerID": "object",
    "Churn": "object",
    "TotalCharges": "float64",
    "tenure": "int64",
    "MonthlyCharges": "float64",
}


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate input dataset before preprocessing and modeling.

    Raises error if validation fails.
    """

    # 1. Required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Target values check
    valid_target_values = {"Yes", "No"}
    observed_values = set(df["Churn"].dropna().unique())

    if not observed_values.issubset(valid_target_values):
        raise ValueError(f"Invalid target values in Churn: {observed_values}")

    # 3. Data type check
    dtype_mismatch = {}
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if expected_dtype not in actual_dtype:
                dtype_mismatch[col] = {
                    "expected": expected_dtype,
                    "actual": actual_dtype
                }

    if dtype_mismatch:
        print("⚠️ Warning: Data type mismatches detected:")
        for col, info in dtype_mismatch.items():
            print(f" - {col}: expected {info['expected']}, got {info['actual']}")

    # 4. Empty dataset check
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty")

    # 5. Basic missing check (warning 수준)
    missing_ratio = df.isnull().mean()

    high_missing = missing_ratio[missing_ratio > 0.3]
    if not high_missing.empty:
        print("⚠️ Warning: High missing ratio detected:")
        print(high_missing)

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("✅ Data validation passed.")