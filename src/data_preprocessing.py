from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import pandas.api.types as ptypes

from .data_inspection import load_datasets

DATA_DIR = Path("data")
OUTPUT_SUFFIX = "cleaned"


def identify_id_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith("_id") or col == "id"]


def identify_date_columns(df: pd.DataFrame) -> list[str]:
    date_like = []
    for col in df.columns:
        low = col.lower()
        if "date" in low or "time" in low or "timestamp" in low:
            date_like.append(col)
    return date_like


def identify_categorical_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    cats = []
    excluded = set(exclude)
    for col in df.columns:
        if col in excluded:
            continue
        if ptypes.is_object_dtype(df[col]) or ptypes.is_categorical_dtype(df[col]):
            cats.append(col)
    return cats


def identify_numeric_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    nums = []
    excluded = set(exclude)
    for col in df.columns:
        if col in excluded:
            continue
        if ptypes.is_numeric_dtype(df[col]) and not ptypes.is_bool_dtype(df[col]):
            nums.append(col)
    return nums


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def drop_missing_ids(df: pd.DataFrame, id_columns: Iterable[str]) -> pd.DataFrame:
    ids = list(id_columns)
    if not ids:
        return df
    return df.dropna(subset=ids)


def fill_missing_categoricals(df: pd.DataFrame, categorical_columns: Iterable[str]) -> pd.DataFrame:
    if not categorical_columns:
        return df
    return df.assign(**{col: df[col].fillna("Unknown") for col in categorical_columns})


def convert_dates(df: pd.DataFrame, date_columns: Iterable[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in date_columns:
        converted[col] = pd.to_datetime(converted[col], errors="coerce")
    return converted


def normalize_numeric(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in numeric_columns:
        series = normalized[col]
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            continue
        normalized[f"{col}_norm"] = (series - min_val) / (max_val - min_val)
    return normalized


def detect_primary_keys(df: pd.DataFrame) -> list[str]:
    pks = []
    for col in identify_id_columns(df):
        series = df[col]
        if series.notna().all() and series.is_unique:
            pks.append(col)
    return pks


def enforce_id_consistency(datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Drop rows whose foreign key values are absent in reference tables."""
    pk_index: Dict[str, Dict[str, set]] = {}
    for name, df in datasets.items():
        pk_index[name] = {}
        for pk in detect_primary_keys(df):
            pk_index[name][pk] = set(df[pk].dropna().unique())

    cleaned: Dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        id_cols = identify_id_columns(df)
        filtered = df.copy()
        for col in id_cols:
            for ref_name, pk_map in pk_index.items():
                if name == ref_name:
                    continue
                if col in pk_map:
                    valid_values = pk_map[col]
                    before = len(filtered)
                    filtered = filtered[filtered[col].isin(valid_values)]
                    after = len(filtered)
                    if before != after:
                        print(f"[id-consistency] {name}.{col}: dropped {before - after} rows not in {ref_name}.{col}")
                    break
        cleaned[name] = filtered
    return cleaned


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = identify_id_columns(df)
    date_cols = identify_date_columns(df)
    cat_cols = identify_categorical_columns(df, exclude=id_cols + date_cols)
    num_cols = identify_numeric_columns(df, exclude=id_cols + date_cols)

    cleaned = remove_duplicates(df)
    cleaned = drop_missing_ids(cleaned, id_cols)
    cleaned = fill_missing_categoricals(cleaned, cat_cols)
    cleaned = convert_dates(cleaned, date_cols)
    cleaned = normalize_numeric(cleaned, num_cols)
    return cleaned


def preprocess_all(data_dir: Path = DATA_DIR, save: bool = True) -> Dict[str, pd.DataFrame]:
    raw = load_datasets(data_dir)
    if not raw:
        print("No datasets loaded. Place CSVs in the data/ directory.")
        return {}

    step1 = {name: clean_dataset(df) for name, df in raw.items()}
    step2 = enforce_id_consistency(step1)

    if save:
        for name, df in step2.items():
            out_path = data_dir / f"{OUTPUT_SUFFIX}_{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"[saved] {out_path}")

    return step2


def main() -> None:
    preprocess_all()


if __name__ == "__main__":
    main()
