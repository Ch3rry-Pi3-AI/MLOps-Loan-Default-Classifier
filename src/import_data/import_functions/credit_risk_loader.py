"""
credit_risk_loader.py

Purpose
-------
Loader/cleaner for the OpenML Credit Risk dataset that arrives in ARFF-like text
with a metadata header and an ``@DATA`` section. This module:
1) Downloads (or reads) the raw text,
2) Strips everything up to and including the ``@DATA`` marker,
3) Parses the rows that follow as CSV,
4) Assigns canonical column names and light dtypes,
5) Optionally saves the cleaned DataFrame to Parquet.
"""
import io
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd # type: ignore
import requests


class CreditRiskLoader:
    """Loader/cleaner for the OpenML Credit Risk dataset."""
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
        """Download, clean, and parse the dataset from a URL."""
        # Step 1: Stream remote text into memory as lines
        lines = self._stream_text_lines(url)
        # Step 2: Extract CSV payload following the '@DATA' marker
        csv_text = self._extract_data_section(lines)
        # Step 3: Convert CSV payload to a typed DataFrame
        return self._to_dataframe(csv_text)

    def from_path(self, path: str | Path) -> pd.DataFrame:
        """Read, clean, and parse the dataset from a local file."""
        # Step 1: Read file contents as text lines (robust to encodings)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        # Step 2: Extract CSV payload
        csv_text = self._extract_data_section(lines)
        # Step 3: Convert to DataFrame
        return self._to_dataframe(csv_text)

    @staticmethod
    def save_parquet(df: pd.DataFrame, out: str | Path) -> Path:
        """Save a DataFrame to Parquet with pyarrow + zstd."""
        # Step 1: Ensure parent directory exists
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Step 2: Write Parquet (compression for smaller files)
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        # Step 3: Return the path for chaining/calling code
        return out

    def _stream_text_lines(self, url: str) -> list[str]:
        """Stream response body and return a list of decoded text lines."""
        # Step 1: Issue HTTP GET with streaming enabled
        r = requests.get(url, stream=True, timeout=self.timeout)
        r.raise_for_status()
        # Step 2: Accumulate text into an in‑memory buffer as it streams
        buf = io.StringIO()
        for chunk in r.iter_content(chunk_size=self.chunk_size):
            # Step 2.1: Skip keep‑alive chunks; decode to UTF‑8 safely
            if chunk:
                buf.write(chunk.decode("utf-8", errors="replace"))
        # Step 3: Reset buffer and split into lines
        buf.seek(0)
        return buf.read().splitlines()

    def _extract_data_section(self, lines: Iterable[str]) -> str:
        """Find the ``@DATA`` marker and return the CSV body as a single string."""
        # Step 1: Locate '@DATA' marker (case-insensitive)
        data_idx: Optional[int] = None
        for i, ln in enumerate(lines):
            if ln.strip().upper().startswith("@DATA"):
                data_idx = i
                break
        # Step 2: Validate presence of '@DATA'
        if data_idx is None:
            raise ValueError("Could not find '@DATA' marker in file.")
        # Step 3: Collect non-empty lines after '@DATA'
        data_lines = [ln for ln in lines[data_idx + 1:] if ln.strip()]
        if not data_lines:
            raise ValueError("No rows found after '@DATA'.")
        # Step 4: Return CSV text for downstream parsing
        return "\n".join(data_lines)

    def _to_dataframe(self, csv_text: str) -> pd.DataFrame:
        """Parse CSV text into a DataFrame and apply light typing."""
        # Step 1: Parse CSV text with no header
        df = pd.read_csv(io.StringIO(csv_text), header=None)
        # Step 2: Validate expected number of columns
        if df.shape[1] != len(self.columns):
            raise ValueError(
                f"Expected {len(self.columns)} columns, found {df.shape[1]}."
            )
        # Step 3: Assign canonical column names
        df.columns = self.columns
        # Step 4: Apply light typing (nullable Int64 for integer-like columns)
        int_cols = ["person_age", "person_income", "loan_amnt", "cb_person_cred_hist_length"]
        float_cols = ["person_emp_length", "loan_int_rate", "loan_percent_income"]

        # Step 4.1: Cast integer-like columns safely
        for c in int_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        # Step 4.2: Cast float-like columns safely
        for c in float_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Step 4.3: Ensure loan_status is integer-like (nullable)
        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce").astype("Int64")

        # Step 5: Return the cleaned DataFrame
        return df