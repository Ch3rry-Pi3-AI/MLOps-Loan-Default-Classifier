from .import_functions.load_params import load_params
from .import_functions.credit_risk_loader import CreditRiskLoader


def import_openml_to_bronze() -> None:

    cfg = load_params("params.json")
    url: str = cfg["import"]["url"]           
    bronze_pq: str = cfg["import"]["output"]  

    loader = CreditRiskLoader()
    df = loader.from_url(url)

    print(df.head().to_string(index=False))

    out = loader.save_parquet(df, bronze_pq)

if __name__ == "__main__":
    import_openml_to_bronze()
