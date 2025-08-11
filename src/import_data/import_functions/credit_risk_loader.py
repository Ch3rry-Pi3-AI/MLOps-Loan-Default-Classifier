import io
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd # type: ignore
import requests


class CreditRiskLoader:
    columns: list[str] = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_status",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
    ]

    def __init__(self, chunk_size: int = 8192, timeout: int = 30):
        self.chunk_size = chunk_size
        self.timeout = timeout

    def from_url(self, url: str) -> pd.DataFrame:


        lines = self._stream_text_lines(url)

        csv_text = self._extract_data_section(lines)

        return self._to_dataframe(csv_text)

    def from_path(self, path: str | Path) -> pd.DataFrame:

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()

        csv_text = self._extract_data_section(lines)

        return self._to_dataframe(csv_text)

    @staticmethod
    def save_parquet(df: pd.DataFrame, out: str | Path) -> Path:

        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)

        return out

    def _stream_text_lines(self, url: str) -> list[str]:

        r = requests.get(url, stream=True, timeout=self.timeout)
        r.raise_for_status()

        buf = io.StringIO()
        for chunk in r.iter_content(chunk_size=self.chunk_size):

            if chunk:
                buf.write(chunk.decode("utf-8", errors="replace"))

        buf.seek(0)
        return buf.read().splitlines()

    def _extract_data_section(self, lines: Iterable[str]) -> str:

        data_idx: Optional[int] = None
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith("@DATA"):
                data_idx = i
                break

        if data_idx is None:
            raise ValueError("Could not find '@DATA' marker in file.")

        data_lines = [ln for ln in lines[data_idx + 1:] if ln.strip()] # type: ignore
        if not data_lines:
            raise ValueError("No rows found after '@DATA'.")
        
        return "\n".join(data_lines)

    def _to_dataframe(self, csv_text: str) -> pd.DataFrame:

        df = pd.read_csv(io.StringIO(csv_text), header=None)

        if df.shape[1] != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} columns, found {df.shape[1]}."
            )
        
        df.columns = self.columns
        int_cols = ["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]
        float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]

        for c in int_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        for c in float_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").astype("Int64")

        return df