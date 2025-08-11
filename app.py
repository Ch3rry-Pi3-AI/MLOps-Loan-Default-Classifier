import os
import json
from pathlib import Path

from src.import_data.import_data_cli import import_openml_to_bronze

def load_params(path: str | Path = "params.json") -> dict:

    with open(path) as f:
        return json.load(f)


def main() -> None:

    params_path = os.getenv("PARAMS_FILE", "params.json")
    cfg = load_params(params_path)

    imp = cfg["import"]

    import_openml_to_bronze()

if __name__ == "__main__":
    main()
