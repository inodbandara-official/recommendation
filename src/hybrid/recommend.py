from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from src.graph_based import recommend_from_similar_users, HeteroGraphRecommender
from src.knowledge_based import KnowledgeMatcher
from src.trend_based import TrendWindowRecommender
from src.data_io import load_dataset
from .hybrid_ranker import HybridRanker, WeightScheme, DEFAULT_TREND_WINDOW_DAYS
from .explanations import attach_explanations


GraphMode = Literal["similarity", "pagerank"]


def _tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [p.strip().strip("'\" ") for p in cleaned.split(",")]
        return {p.lower() for p in parts if p}
    if isinstance(val, (list, tuple, set)):
        return {str(v).strip().lower() for v in val if pd.notna(v)}
    return {str(val).strip().lower()}


def _filter_future_events(events: pd.DataFrame, today: Optional[pd.Timestamp]) -> pd.DataFrame:
    if today is None or "date" not in events.columns:
        return events
    parsed = pd.to_datetime(events["date"], errors="coerce")
    mask = parsed.isna() | (parsed >= today)
    return events.loc[mask].copy()


_PAGERANK_CACHE: dict[tuple, HeteroGraphRecommender] = {}


def _frame_fingerprint(*frames: pd.DataFrame) -> tuple:
    """Cheap content fingerprint for cache invalidation."""
    parts = []
    for df in frames:
        if df is None or df.empty:
            parts.append((0, 0))
        else:
            parts.append((len(df), tuple(df.columns)))
    return tuple(parts)


def _build_pagerank_recommender(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    events: pd.DataFrame,
) -> HeteroGraphRecommender:
    key = _frame_fingerprint(attends, follows, events)
    cached = _PAGERANK_CACHE.get(key)
    if cached is not None:
        return cached

    rec = HeteroGraphRecommender()
    expanded_rows = []
    if "event_id" in events.columns and "artist_ids" in events.columns:
        for _, row in events[["event_id", "artist_ids"]].iterrows():
            ev = row["event_id"]
            artists = row["artist_ids"]
            if isinstance(artists, str):
                artists = _tokens(artists)
            if isinstance(artists, (list, tuple, set)):
                for a in artists:
                    expanded_rows.append({"event_id": ev, "artist_id": a})
    expanded = pd.DataFrame(expanded_rows) if expanded_rows else None
    rec.build_from_frames(attends=attends, follows=follows, events=expanded)
    _PAGERANK_CACHE[key] = rec
    return rec


def _graph_pagerank(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    events: pd.DataFrame,
    user_id: str,
    top_n: int,
) -> pd.DataFrame:
    rec = _build_pagerank_recommender(attends, follows, events)
    pairs = rec.recommend_events_for_user(user_id, top_k=top_n)
    if not pairs:
        return pd.DataFrame(columns=["event_id", "GraphScore"])
    rows = [(node[len("event:"):], score) for node, score in pairs]
    return pd.DataFrame(rows, columns=["event_id", "GraphScore"])


def recommend_events(
    user_id: str,
    top_n: int = 10,
    data_dir: Path = Path("data"),
    graph_mode: GraphMode = "similarity",
    trend_window_days: int = DEFAULT_TREND_WINDOW_DAYS,
    today: Optional[pd.Timestamp | str] = None,
    future_only: bool = True,
    weights: dict | None = None,
    interaction_threshold: int = 5,
) -> pd.DataFrame:
    """Generate hybrid recommendations with explanations.

    Args:
        graph_mode: "similarity" (vectorized cosine + Adamic-Adar) or "pagerank"
            (heterogeneous user-event-artist personalized PageRank).
        trend_window_days: trend recency window.
        today: reference date for future-event filtering.
        future_only: drop events whose date is before `today`.
        weights: optional override for HybridRanker strategy weights.
    """
    ds = load_dataset(data_dir)
    users = ds["users"]
    events = ds["events"]
    attends = ds["attends"]
    follows = ds["follows"]

    if users.empty or events.empty:
        raise FileNotFoundError("Users and events data are required.")

    if attends.empty:
        attends = pd.DataFrame(columns=["user_id", "event_id", "timestamp"])
    if follows.empty:
        follows = pd.DataFrame(columns=["user_id", "artist_id", "timestamp"])

    # Step 8: filter to future events for scoring
    if future_only:
        ref_today = pd.to_datetime(today) if today is not None else pd.Timestamp.utcnow().normalize()
        events_for_scoring = _filter_future_events(events, ref_today)
        if events_for_scoring.empty:
            events_for_scoring = events  # fall back if everything is in the past
    else:
        events_for_scoring = events

    # Knowledge-based scores
    km = KnowledgeMatcher(budget_col=None)
    km.fit(users, events_for_scoring)
    knowledge_df = km.recommend(user_id, top_n=len(events_for_scoring))
    knowledge_scores = knowledge_df[["event_id", "KnowledgeScore"]]

    # Graph-based scores
    if graph_mode == "pagerank":
        graph_scores = _graph_pagerank(
            attends, follows, events_for_scoring, user_id, top_n=max(top_n * 5, 50)
        )
    else:
        graph_df = recommend_from_similar_users(
            attends=attends,
            follows=follows,
            target_user=user_id,
            top_users=50,
            top_n=max(top_n * 5, 50),
            alpha=0.5,
        )
        graph_scores = graph_df if not graph_df.empty else pd.DataFrame(columns=["event_id", "GraphScore"])

    # Trend-based scores
    if attends.empty:
        trend_scores = pd.DataFrame(columns=["event_id", "TrendScore"])
    else:
        trend_model = TrendWindowRecommender().fit(attends)
        trend_df = trend_model.recommend(top_n=max(top_n * 5, 50), window_days=trend_window_days)
        trend_scores = trend_df[["event_id", "TrendScore"]] if not trend_df.empty else pd.DataFrame(columns=["event_id", "TrendScore"])

    # Restrict candidate pool to scoring set (future-only if applicable)
    candidate_ids = set(events_for_scoring["event_id"].astype(str))
    knowledge_scores = knowledge_scores[knowledge_scores["event_id"].astype(str).isin(candidate_ids)]
    graph_scores = graph_scores[graph_scores["event_id"].astype(str).isin(candidate_ids)]
    trend_scores = trend_scores[trend_scores["event_id"].astype(str).isin(candidate_ids)]

    # Merge scores
    candidates = pd.DataFrame({"event_id": pd.unique(
        pd.concat([
            knowledge_scores["event_id"],
            graph_scores.get("event_id", pd.Series(dtype=str)),
            trend_scores.get("event_id", pd.Series(dtype=str)),
        ], ignore_index=True)
    )})

    if candidates.empty:
        return pd.DataFrame(columns=["event_id", "KnowledgeScore", "GraphScore", "TrendScore", "FinalScore", "Explanations"])

    merged = candidates.merge(knowledge_scores, on="event_id", how="left")
    merged = merged.merge(graph_scores, on="event_id", how="left")
    merged = merged.merge(trend_scores, on="event_id", how="left")
    for col in ("KnowledgeScore", "GraphScore", "TrendScore"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # Hybrid ranking with activity-level gate
    user_row = users.loc[users["user_id"] == user_id].head(1)
    activity_level = (
        str(user_row.iloc[0]["activity_level"])
        if not user_row.empty and "activity_level" in user_row.columns
        else None
    )
    user_interactions = len(attends.loc[attends["user_id"] == user_id])
    ranker_weights = None
    if weights is not None:
        ranker_weights = {k: WeightScheme(**v) if isinstance(v, dict) else v for k, v in weights.items()}
    ranker = HybridRanker(interaction_threshold=interaction_threshold, weights=ranker_weights)
    ranked = ranker.rank(
        merged,
        user_interactions=user_interactions,
        top_n=top_n,
        activity_level=activity_level,
    )

    # Explanations (use the full event metadata for descriptions)
    interests = _tokens(user_row.iloc[0]["art_interests"]) if not user_row.empty and "art_interests" in user_row.columns else None
    city = user_row.iloc[0]["city"] if not user_row.empty and "city" in user_row.columns else None
    ranked = attach_explanations(ranked, events=events, user_interests=interests, user_city=city)
    return ranked
