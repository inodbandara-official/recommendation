"""Central JSON dataset loader.

The Recommendation project's dataset is now a single JSON file
(`rasaswadaya_large_dataset.json`) produced by the GNN pipeline. This module
loads it once and exposes per-table DataFrames (users / artists / events /
attends / follows) with the same column names the rest of the codebase
already expects.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

import pandas as pd

JSON_FILENAME = "rasaswadaya_large_dataset.json"
DEFAULT_DATA_DIR = Path("data")


def _resolve_path(data_dir: Path | str | None) -> Path:
    return Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR


@lru_cache(maxsize=4)
def _load_raw(data_dir_str: str) -> dict:
    path = Path(data_dir_str) / JSON_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found at {path}. Expected {JSON_FILENAME} in {data_dir_str}/."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_users(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw.get("users", []))
    if df.empty:
        return df
    # Provide singular `language` column derived from list (legacy code expects str)
    if "language_preferences" in df.columns and "language" not in df.columns:
        df["language"] = df["language_preferences"].apply(
            lambda v: v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
        )
    # Aliases the legacy KnowledgeMatcher / hybrid code may look for
    if "interests" not in df.columns and "art_interests" in df.columns:
        df["interests"] = df["art_interests"]
    if "moods" not in df.columns and "mood_preferences" in df.columns:
        df["moods"] = df["mood_preferences"]
    if "genres" not in df.columns and "culture_preferences" in df.columns:
        df["genres"] = df["culture_preferences"]
    return df


def _build_artists(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw.get("artists", []))
    if df.empty:
        return df
    # Original schema had both `language` (str) and `languages` (list); JSON has
    # `language` as a list. Normalize: keep `languages` (list), make `language` str.
    if "language" in df.columns:
        sample = df["language"].dropna().head(1).tolist()
        if sample and isinstance(sample[0], list):
            df["languages"] = df["language"]
            df["language"] = df["language"].apply(
                lambda v: v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
            )
    # Provide `art_form` (singular) for any legacy code path expecting it
    if "art_form" not in df.columns and "art_forms" in df.columns:
        df["art_form"] = df["art_forms"].apply(
            lambda v: v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
        )
    return df


def _build_events(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame(raw.get("events", []))
    if df.empty:
        return df
    # Split `date` (ISO timestamp) into legacy `date` (YYYY-MM-DD) + `time` (HH:MM)
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        if "time" not in df.columns:
            df["time"] = parsed.dt.strftime("%H:%M").fillna("")
        df["date"] = parsed.dt.strftime("%Y-%m-%d").fillna(df["date"].astype(str))
    return df


def _build_interactions(raw: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(raw.get("interactions", {}).get(key, []))


def load_users(data_dir: Path | str | None = None) -> pd.DataFrame:
    return _build_users(_load_raw(str(_resolve_path(data_dir))))


def load_artists(data_dir: Path | str | None = None) -> pd.DataFrame:
    return _build_artists(_load_raw(str(_resolve_path(data_dir))))


def load_events(data_dir: Path | str | None = None) -> pd.DataFrame:
    return _build_events(_load_raw(str(_resolve_path(data_dir))))


def load_attends(data_dir: Path | str | None = None) -> pd.DataFrame:
    return _build_interactions(_load_raw(str(_resolve_path(data_dir))), "attends")


def load_follows(data_dir: Path | str | None = None) -> pd.DataFrame:
    return _build_interactions(_load_raw(str(_resolve_path(data_dir))), "follows")


def load_dataset(data_dir: Path | str | None = None) -> Dict[str, pd.DataFrame]:
    raw = _load_raw(str(_resolve_path(data_dir)))
    return {
        "users": _build_users(raw),
        "artists": _build_artists(raw),
        "events": _build_events(raw),
        "attends": _build_interactions(raw, "attends"),
        "follows": _build_interactions(raw, "follows"),
    }
