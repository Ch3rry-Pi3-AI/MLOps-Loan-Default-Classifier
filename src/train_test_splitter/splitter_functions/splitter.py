"""
splitter.py

Purpose
-------
Create train/test Parquet datasets from the Silver dataset, with optional
stratification on a given column (commonly the classification target).
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 0: Ensure pyarrow is available for Parquet I/O early
try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("❌ pyarrow is required. Install it with: pip install pyarrow")


def create_train_test_datasets(
    silver_parquet: Path,
    train_parquet: Path,
    test_parquet: Path,
    test_size: float,
    random_state: int,
    stratify_col: str | None = None,
) -> tuple[Path, Path]:
    """Split a Silver Parquet dataset into train/test Parquets (Gold layer)."""
    # Step 1: Load the Silver dataset
    df = pd.read_parquet(silver_parquet, engine="pyarrow")
    print(f"🥈 Loaded silver data: {df.shape[0]} rows, {df.shape[1]} columns...\n")

    # Step 2: Prepare optional stratification vector
    stratify = None
    if stratify_col:
        # Step 2.1: Validate stratify column presence
        if stratify_col not in df.columns:
            raise KeyError(f"❌ Stratify column '{stratify_col}' not found in data columns")
        stratify = df[stratify_col]
        print(f"Stratifying split by column: '{stratify_col}'...\n")

    # Step 3: Perform the split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Step 4: Ensure output directories exist
    train_parquet.parent.mkdir(parents=True, exist_ok=True)
    test_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Step 5: Write Parquet outputs
    train_df.to_parquet(train_parquet, index=False, engine="pyarrow", compression="zstd")
    test_df.to_parquet(test_parquet, index=False, engine="pyarrow", compression="zstd")

    # Step 6: Report
    print(f"🥇 Wrote train set: {train_df.shape[0]} rows ({len(train_df)/len(df):.1%}) → {train_parquet}...\n")
    print(f"🥇 Wrote  test set: {test_df.shape[0]} rows ({len(test_df)/len(df):.1%}) → {test_parquet}...\n")

    # Step 7: Return paths for downstream steps
    return train_parquet, test_parquet
