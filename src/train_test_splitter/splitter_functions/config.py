"""
config.py

Purpose
-------
Typed configuration for the train/test split step.
"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class TrainTestSplitConfig:
    """Immutable configuration for train/test splitting."""
    input: Path
    train_output: Path
    test_output: Path
    test_size: float
    random_state: int
    stratify_col: str

    @staticmethod
    def from_params(params_path: str | Path = "params.json") -> "TrainTestSplitConfig":
        """Create config by reading the ``split`` block from a JSON params file."""
        # Step 1: Open and parse JSON
        with open(params_path) as f:
            cfg = json.load(f)

        # Step 2: Retrieve the 'split' block
        sp = cfg.get("split")
        if sp is None:
            raise KeyError("Missing 'split' block in params.json")

        # Step 3: Validate required keys
        required = ("input", "train_output", "test_output", "target_col", "test_size", "random_state")
        missing = [k for k in required if k not in sp]
        if missing:
            raise KeyError(f"Missing split.{', '.join(missing)} in params.json")

        # Step 4: Construct frozen config
        return TrainTestSplitConfig(
            input=Path(sp["input"]),
            train_output=Path(sp["train_output"]),
            test_output=Path(sp["test_output"]),
            test_size=float(sp["test_size"]),
            random_state=int(sp["random_state"]),
            stratify_col=sp["target_col"] 
        )
