"""
splitter_cli.py

Purpose
-------
Command-line entrypoint that:
1) Reads split configuration from ``params.json`` (or ``$PARAMS_FILE``),
2) Calls the splitter to produce train/test Parquet outputs.
"""

import os
from .splitter_functions.config import TrainTestSplitConfig
from .splitter_functions.splitter import create_train_test_datasets


def main() -> None:
    """Resolve config and execute the train/test split."""
    # Step 1: Resolve the params path (allow env override)
    params_path = os.getenv("PARAMS_FILE", "params.json")

    # Step 2: Parse configuration
    cfg = TrainTestSplitConfig.from_params(params_path)

    # Step 3: Execute the split using the resolved config
    create_train_test_datasets(
        silver_parquet=cfg.input,
        train_parquet=cfg.train_output,
        test_parquet=cfg.test_output,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify_col=cfg.stratify_col
    )


if __name__ == "__main__":
    # Step 0: Allow running as a script: `python -m src.train_test_splitter.splitter_cli`
    main()