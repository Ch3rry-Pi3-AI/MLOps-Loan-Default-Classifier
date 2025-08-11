from pathlib import Path
import json


def load_params(path: str | Path = "params.json") -> dict:
    with open(path) as f:
        return json.load(f)