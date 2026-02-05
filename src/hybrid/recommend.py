from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.graph_based import recommend_from_similar_users
from src.knowledge_based import KnowledgeMatcher
from src.trend_based import TrendWindowRecommender
from .hybrid_ranker import HybridRanker
from .explanations import attach_explanations


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _load_csv_prefer_cleaned(data_dir: Path, name: str) -> Optional[pd.DataFrame]:
    cleaned = _load_csv(data_dir / f"cleaned_{name}.csv")
    if cleaned is not None:
        return cleaned
    return _load_csv(data_dir / f"{name}.csv")


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


def recommend_events(user_id: str, top_n: int = 10, data_dir: Path = Path("data")) -> pd.DataFrame:
    """Generate hybrid recommendations with explanations.

    Loads cleaned CSVs if present, otherwise raw CSVs.
    """
    users = _load_csv_prefer_cleaned(data_dir, "users")
    events = _load_csv_prefer_cleaned(data_dir, "events")
    attends = _load_csv_prefer_cleaned(data_dir, "attends")
    follows = _load_csv_prefer_cleaned(data_dir, "follows")

    if users is None or events is None:
        raise FileNotFoundError("Users and events data are required.")

    if attends is None:
        attends = pd.DataFrame(columns=["user_id", "event_id", "timestamp"])
    if follows is None:
        follows = pd.DataFrame(columns=["user_id", "artist_id", "timestamp"])

    # Knowledge-based scores
    km = KnowledgeMatcher(budget_col=None)
    km.fit(users, events)
    knowledge_df = km.recommend(user_id, top_n=len(events))
    knowledge_scores = knowledge_df[["event_id", "KnowledgeScore"]]

    # Graph-based scores
    graph_df = recommend_from_similar_users(
        attends=attends,
        follows=follows,
        target_user=user_id,
        top_users=50,
        top_n=max(top_n * 3, top_n),
        alpha=0.5,
    )
    if graph_df.empty:
        graph_scores = pd.DataFrame(columns=["event_id", "GraphScore"])
    else:
        graph_scores = graph_df.rename(columns={"GraphScore": "GraphScore"})

    # Trend-based scores
    if attends.empty:
        trend_scores = pd.DataFrame(columns=["event_id", "TrendScore"])
    else:
        trend_model = TrendWindowRecommender().fit(attends)
        trend_df = trend_model.recommend(top_n=max(top_n * 3, top_n), window_days=14)
        trend_scores = trend_df[["event_id", "TrendScore"]]

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
    merged[["KnowledgeScore", "GraphScore", "TrendScore"]] = merged[
        ["KnowledgeScore", "GraphScore", "TrendScore"]
    ].fillna(0.0)

    # Hybrid ranking
    user_interactions = len(attends.loc[attends["user_id"] == user_id])
    ranker = HybridRanker()
    ranked = ranker.rank(merged, user_interactions=user_interactions, top_n=top_n)

    # Explanations
    user_row = users.loc[users["user_id"] == user_id].head(1)
    interests = _tokens(user_row.iloc[0]["art_interests"]) if not user_row.empty and "art_interests" in user_row else None
    region = user_row.iloc[0]["region_preference"] if not user_row.empty and "region_preference" in user_row else None
    ranked = attach_explanations(ranked, events=events, user_interests=interests, user_region=region)
    return ranked
