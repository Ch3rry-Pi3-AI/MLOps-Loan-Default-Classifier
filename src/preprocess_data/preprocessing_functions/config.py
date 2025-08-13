"""
config.py

Purpose
-------
Typed configuration holder for the preprocessing pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable configuration for the preprocessing step."""

    input: Path
    output: Path
    scaler: str = "standard"          
    target_col: str = "loan_status"
    poly_degree: int = 2
    save_artifacts_dir: Path | None = None  

    @staticmethod
    def from_params(params_path: str | Path = "params.json") -> "PreprocessConfig":
        """Create a ``PreprocessConfig`` by reading a JSON params file."""
        # Step 1: Read the JSON file
        with open(params_path) as f:
            cfg = json.load(f)

        # Step 2: Extract the 'preprocess' block
        pre = cfg.get("preprocess")
        if pre is None:
            raise KeyError("Missing 'preprocess' block in params.json")

        # Step 3: Validate required keys
        missing = [k for k in ("input", "output") if k not in pre]
        if missing:
            raise KeyError(f"Missing preprocess.{', '.join(missing)} in params.json")

        # Step 4: Construct the frozen config
        return PreprocessConfig(
            input=Path(pre["input"]),
            output=Path(pre["output"]),
            scaler=pre.get("scaler", "standard"),
            target_col=pre.get("target_col", "loan_status"),
            poly_degree=int(pre.get("poly_degree", 2)),
            save_artifacts_dir=Path(pre["save_artifacts_dir"]) if pre.get("save_artifacts_dir") else None,
        )
