from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_interactions(csv_path: str | Path, expected_columns: Iterable[str] | None = None, **read_csv_kwargs) -> pd.DataFrame:
    """Load interactions data from CSV and optionally validate expected columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {path}")

    df = pd.read_csv(path, **read_csv_kwargs)
    if expected_columns:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
    return df
