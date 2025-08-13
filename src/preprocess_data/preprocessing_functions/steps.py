"""
steps.py

Purpose
-------
Composable, unit-testable preprocessing steps used to transform the Bronze dataset
into a Silver dataset: I/O, validation, imputation, encoding, scaling, polynomial
expansion, and final write.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

# Expected schema for transformations
NUMERIC_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def read_bronze(path: Path) -> pd.DataFrame:
    """Read Bronze Parquet into a DataFrame."""
    # Step 1: Validate existence
    if not path.is_file():
        raise FileNotFoundError(f"Input Parquet not found: {path}")
    # Step 2: Read Parquet
    df = pd.read_parquet(path, engine="pyarrow")
    # Step 3: Basic info
    print(f"\n🥉 Bronze data loaded: {df.shape[0]} rows, {df.shape[1]} columns...\n")
    return df


def split_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target column from features."""
    # Step 1: Validate presence
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in bronze data")
    # Step 2: Separate
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def validate_columns(df: pd.DataFrame,
                     numeric_cols: list[str] = NUMERIC_COLS,
                     categorical_cols: list[str] = CATEGORICAL_COLS) -> None:
    """Verify expected numeric and categorical columns are present."""
    # Step 1: Compute missing sets
    missing_numeric = [c for c in numeric_cols if c not in df.columns]
    missing_cats = [c for c in categorical_cols if c not in df.columns]
    # Step 2: Raise if any missing
    if missing_numeric or missing_cats:
        raise KeyError(
            "Missing expected columns. "
            f"Numeric missing: {missing_numeric}; Categorical missing: {missing_cats}"
        )


def impute_numerics(df: pd.DataFrame, numeric_cols: list[str] = NUMERIC_COLS) -> pd.DataFrame:
    """Impute missing numeric values with column medians."""
    # Step 1: Work on a copy to avoid side effects
    df = df.copy()
    # Step 2: Median imputation per numeric column
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str] = CATEGORICAL_COLS) -> pd.DataFrame:
    """One-hot encode categorical columns (drop first to avoid collinearity)."""
    # Step 1: Delegate to pandas.get_dummies
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)


def make_scaler(kind: str):
    """Factory returning an appropriate scaler."""
    # Step 1: Branch by requested kind
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler_type must be 'standard' or 'minmax'")


def scale_numerics(df: pd.DataFrame, numeric_cols: list[str], scaler) -> tuple[pd.DataFrame, object]:
    """Scale numeric columns using the provided scaler (fit + transform)."""
    # Step 1: Copy to avoid mutating caller’s DataFrame
    df = df.copy()
    # Step 2: Fit + transform the numeric slice
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    # Step 2.1: Cast to float32 for compactness
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    return df, scaler


def poly_expand(df: pd.DataFrame, numeric_cols: list[str], degree: int) -> tuple[pd.DataFrame, PolynomialFeatures]:
    """Generate polynomial features from numeric columns."""
    # Step 1: Fit PolynomialFeatures on numeric slice
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    arr = poly.fit_transform(df[numeric_cols])
    names = poly.get_feature_names_out(numeric_cols)

    # Step 2: Construct expanded DataFrame aligned to original index
    poly_df = pd.DataFrame(arr, columns=names, index=df.index)

    # Step 3: Drop original numeric columns and concatenate expanded ones
    df_ = df.drop(columns=numeric_cols)
    df_ = pd.concat([df_, poly_df], axis=1)
    return df_, poly


def write_silver(df: pd.DataFrame, path: Path) -> None:
    """Write the transformed dataset to Silver Parquet."""
    # Step 1: Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Step 2: Write Parquet with compression
    df.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    # Step 3: Confirmation
    print(f"🥈 Preprocessed data written to {path} ({df.shape[0]} rows, {df.shape[1]} cols)...\n")
