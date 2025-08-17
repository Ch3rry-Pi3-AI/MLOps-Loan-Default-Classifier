import os
import json
from pathlib import Path

from src.import_data.import_data_cli import import_openml_to_bronze
from src.preprocess_data.feature_engineering_cli import feature_engineering

def load_params(path: str | Path = "params.json") -> dict:

    with open(path) as f:
        return json.load(f)


def main() -> None:

    params_path = os.getenv("PARAMS_FILE", "params.json")
    cfg = load_params(params_path)

    imp = cfg["import"]
    pre = cfg["preprocess"]

    import_openml_to_bronze()
    feature_engineering()

if __name__ == "__main__":
    main()
