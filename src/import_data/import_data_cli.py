"""
import_data_cli.py

Purpose
-------
Command-line entrypoint that orchestrates the "import" step of the pipeline:
1) Read configuration (URL and output path) from ``params.json``,
2) Download and clean the OpenML Credit Risk dataset,
3) Persist a Bronze Parquet file to the configured location.
"""
from .import_functions.load_params import load_params
from .import_functions.credit_risk_loader import CreditRiskLoader


def import_openml_to_bronze() -> None:
    # Step 1: Load configuration
    cfg = load_params("params.json")
    url: str = cfg["import"]["url"]           
    bronze_pq: str = cfg["import"]["output"]

    # Step 2: Download + clean with the loader
    loader = CreditRiskLoader()
    df = loader.from_url(url)
    # Step 2.1: Basic sanity feedback
    print(df.head().to_string(index=False))
    # Step 3: Persist Bronze Parquet
    out = loader.save_parquet(df, bronze_pq)

if __name__ == "__main__":
    # Step 4: Execute as a script
    import_openml_to_bronze()
