"""
run.py

Purpose
-------
CLI-style orchestrator for the preprocessing pipeline. It:
1) Loads configuration (``PreprocessConfig``),
2) Reads Bronze, splits target, validates schema,
3) Imputes numerics, one-hot encodes categoricals,
4) Scales numerics, performs polynomial expansion,
5) Reattaches target and writes Silver,
6) Optionally saves artefacts (scaler, poly, schema) for inference.
"""

import os
from pathlib import Path
import joblib  
from .preprocessing_functions.config import PreprocessConfig
from .preprocessing_functions.steps import (
    read_bronze, split_target, validate_columns, impute_numerics,
    one_hot_encode, make_scaler, scale_numerics, poly_expand, write_silver,
    NUMERIC_COLS, CATEGORICAL_COLS
)


def run_preprocess(cfg: PreprocessConfig) -> Path:
    """Execute the preprocessing pipeline using the provided config."""
    # Step 1: Load Bronze
    df = read_bronze(cfg.input)

    # Step 2: Split target and validate schema
    X, y = split_target(df, cfg.target_col)
    validate_columns(X, NUMERIC_COLS, CATEGORICAL_COLS)

    # Step 3: Impute numeric NaNs
    X = impute_numerics(X, NUMERIC_COLS)

    # Step 4: One‑hot encode categoricals
    X = one_hot_encode(X, CATEGORICAL_COLS)

    # Step 5: Scale numerics
    scaler = make_scaler(cfg.scaler)
    X, scaler = scale_numerics(X, NUMERIC_COLS, scaler)

    # Step 6: Polynomial expansion
    X, poly = poly_expand(X, NUMERIC_COLS, cfg.poly_degree)

    # Step 7: Reattach target and write Silver
    X[cfg.target_col] = y
    write_silver(X, cfg.output)

    # Step 8: Optionally persist artefacts for inference
    if cfg.save_artifacts_dir:
        cfg.save_artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, cfg.save_artifacts_dir / "scaler.joblib")
        joblib.dump(poly,   cfg.save_artifacts_dir / "poly.joblib")
        joblib.dump({"numeric_cols": NUMERIC_COLS}, cfg.save_artifacts_dir / "schema.joblib")
        print(f"Saved artefacts to {cfg.save_artifacts_dir}")

    return cfg.output


def feature_engineering() -> None:
    """Convenience wrapper to run the pipeline using an environment-driven params path."""
    # Step 1: Resolve params path (env override supported)
    params_path = os.getenv("PARAMS_FILE", "params.json")

    # Step 2: Parse config and run pipeline
    cfg = PreprocessConfig.from_params(params_path)
    run_preprocess(cfg)


if __name__ == "__main__":
    # Step 0: Entry point for ``python -m src.preprocess.run``
    feature_engineering()
