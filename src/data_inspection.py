from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

DATA_FILES = {
    "users": "users.csv",
    "events": "events.csv",
    "artists": "artists.csv",
    "attends": "attends.csv",
    "follows": "follows.csv",
}


def load_datasets(data_dir: Path = Path("data")) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for name, filename in DATA_FILES.items():
        path = data_dir / filename
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        datasets[name] = df
    return datasets


def detect_primary_keys(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if col.endswith("_id") or col == "id":
            series = df[col]
            if series.notna().all() and series.is_unique:
                candidates.append(col)
    return candidates


def detect_foreign_keys(name: str, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for col in df.columns:
        if not col.endswith("_id"):
            continue
        if col in detect_primary_keys(df):
            continue
        base = col[:-3]  # drop _id
        target_names = [base, f"{base}s", f"{base}es"]
        for target in target_names:
            target_df = all_data.get(target)
            if target_df is None or col not in target_df.columns:
                continue
            fk_values = set(df[col].dropna().unique())
            pk_values = set(target_df[col].dropna().unique())
            if fk_values and fk_values.issubset(pk_values):
                results.append((col, target))
                break
    return results


def print_basic_report(name: str, df: pd.DataFrame, all_data: Dict[str, pd.DataFrame]) -> None:
    print(f"\n=== {name} ===")
    print("Columns:", list(df.columns))
    print("First 5 rows:")
    print(df.head(5))
    print("Row count:", len(df))

    pks = detect_primary_keys(df)
    if pks:
        print("Primary key candidates:", pks)
    else:
        print("Primary key candidates: none detected")

    fks = detect_foreign_keys(name, df, all_data)
    if fks:
        fk_text = [f"{col} -> {target}" for col, target in fks]
        print("Foreign key candidates:", fk_text)
    else:
        print("Foreign key candidates: none detected")


def main() -> None:
    datasets = load_datasets()
    if not datasets:
        print("No datasets loaded. Place CSVs in the data/ directory.")
        return

    for name, df in datasets.items():
        print_basic_report(name, df, datasets)


if __name__ == "__main__":
    main()
