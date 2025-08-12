"""
load_params.py

Purpose
-------
Utility for loading project configuration from a JSON file (default: ``params.json``).
"""
from pathlib import Path
import json


def load_params(path: str | Path = "params.json") -> dict:
    with open(path) as f:
        return json.load(f)