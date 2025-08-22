import os
import json
from pathlib import Path

from src.import_data.import_data_cli import import_openml_to_bronze
from src.preprocess_data.feature_engineering_cli import feature_engineering
from src.train_test_splitter.splitter_functions.config import TrainTestSplitConfig
from src.train_test_splitter.splitter_cli import create_train_test_datasets

cfg_split = TrainTestSplitConfig.from_params("params.json")

def load_params(path: str | Path = "params.json") -> dict:

    with open(path) as f:
        return json.load(f)


def main() -> None:

    params_path = os.getenv("PARAMS_FILE", "params.json")
    cfg = load_params(params_path)

    imp = cfg["import"]
    pre = cfg["preprocess"]
    sp  = cfg["split"]

    import_openml_to_bronze()
    feature_engineering()
    create_train_test_datasets(
        silver_parquet=cfg_split.input,
        train_parquet=cfg_split.train_output,
        test_parquet=cfg_split.test_output,
        test_size=cfg_split.test_size,
        random_state=cfg_split.random_state,
        stratify_col=cfg_split.stratify_col
    )

if __name__ == "__main__":
    main()
