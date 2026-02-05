from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


def _to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [p.strip().strip("'\" ") for p in cleaned.split(",")]
        return {p.lower() for p in parts if p}
    if isinstance(val, Iterable) and not isinstance(val, (bytes, bytearray)):
        return {str(v).strip().lower() for v in val if pd.notna(v)}
    return {str(val).strip().lower()}


def attach_explanations(
    recommendations: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
    user_interests: Optional[Iterable[str]] = None,
    user_region: Optional[str] = None,
) -> pd.DataFrame:
    """Attach human-readable reasons to each recommended event.

    recommendations: DataFrame with at least 'event_id' and score columns (KnowledgeScore, GraphScore, TrendScore).
    events: optional DataFrame with event metadata (art_forms, genres, region) keyed by event_id.
    """
    if "event_id" not in recommendations:
        raise ValueError("recommendations must include 'event_id'")

    interests_tokens = {i.strip().lower() for i in user_interests} if user_interests else set()
    user_region_norm = user_region.strip().lower() if user_region else None

    event_meta = None
    if events is not None and "event_id" in events.columns:
        event_meta = events.set_index("event_id")

    def reason_for_row(row: pd.Series) -> list[str]:
        reasons: list[str] = []

        # Interest match via KnowledgeScore and event metadata
        if event_meta is not None and row["event_id"] in event_meta.index and interests_tokens:
            meta_row = event_meta.loc[row["event_id"]]
            tokens = set()
            for col in ("art_forms", "genres"):
                if col in meta_row:
                    tokens.update(_to_tokens(meta_row[col]))
            if tokens and interests_tokens.intersection(tokens):
                reasons.append("Matches your interests")

        # Graph-based social proof
        if row.get("GraphScore", 0) > 0:
            reasons.append("Popular among similar users")

        # Trending signal with regional hint if present
        if row.get("TrendScore", 0) > 0:
            if event_meta is not None and row["event_id"] in event_meta.index:
                meta_row = event_meta.loc[row["event_id"]]
                region_val = meta_row.get("region") if isinstance(meta_row, pd.Series) else None
                region_token = None
                if pd.notna(region_val):
                    region_token = str(region_val).strip().lower()
                if user_region_norm and region_token and user_region_norm == region_token:
                    reasons.append("Trending this week near you")
                else:
                    reasons.append("Trending this week")
            else:
                reasons.append("Trending this week")

        # Ensure at least one reason
        if not reasons:
            reasons.append("Recommended based on combined scores")
        return reasons

    result = recommendations.copy()
    result["Explanations"] = result.apply(reason_for_row, axis=1)
    return result
