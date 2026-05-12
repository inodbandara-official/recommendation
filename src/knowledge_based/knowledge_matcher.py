from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def _to_tokens(val: object) -> set[str]:
    """Normalize a cell value into a lowercase token set."""
    if val is None:
        return set()
    if isinstance(val, float) and pd.isna(val):
        return set()
    if isinstance(val, (list, tuple, set)):
        out: set[str] = set()
        for v in val:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except (TypeError, ValueError):
                pass
            s = str(v).strip().lower()
            if s:
                out.add(s)
        return out
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip("'\" ").lower() for p in s.split(",")]
        return {p for p in parts if p}
    return {str(val).strip().lower()}


class KnowledgeMatcher:
    """Knowledge-based recommender with graded multi-field matching.

    Uses Jaccard-style overlap counts (not binary) across multiple user/event
    fields. Honors `activity_level` for budget proxy and adds a popularity
    prior so brand-new low-attendance events aren't tied at zero.
    """

    DEFAULT_FIELD_PAIRS: tuple[tuple[str, str, float], ...] = (
        # (user_field, event_field, weight)
        ("art_interests", "art_forms", 0.25),
        ("culture_preferences", "genres", 0.20),
        ("mood_preferences", "moods", 0.20),
        ("language_preferences", "languages", 0.10),
        ("city", "city", 0.10),
    )
    BUDGET_WEIGHT = 0.10
    POPULARITY_WEIGHT = 0.05

    # Used when user has no explicit budget
    ACTIVITY_BUDGET_PERCENTILE = {"low": 25, "medium": 50, "high": 75}

    def __init__(
        self,
        user_id_col: str = "user_id",
        price_col: str = "ticket_price",
        budget_col: Optional[str] = None,
        field_pairs: Sequence[tuple[str, str, float]] | None = None,
    ) -> None:
        self.user_id_col = user_id_col
        self.price_col = price_col
        self.budget_col = budget_col
        self.field_pairs = tuple(field_pairs) if field_pairs is not None else self.DEFAULT_FIELD_PAIRS
        self.users: Optional[pd.DataFrame] = None
        self.events: Optional[pd.DataFrame] = None
        self._event_tokens: dict[str, list[set[str]]] = {}
        self._popularity: Optional[np.ndarray] = None

    def fit(self, users: pd.DataFrame, events: pd.DataFrame) -> "KnowledgeMatcher":
        if self.user_id_col not in users.columns:
            raise ValueError(f"Users data missing required column: {self.user_id_col}")
        if "event_id" not in events.columns:
            raise ValueError("Events data missing required column: event_id")

        self.users = users.copy()
        self.events = events.copy().reset_index(drop=True)

        # Pre-tokenize event fields once (vectorization key)
        self._event_tokens = {}
        for _, event_field, _ in self.field_pairs:
            if event_field in self.events.columns:
                self._event_tokens[event_field] = [
                    _to_tokens(v) for v in self.events[event_field].tolist()
                ]
            else:
                self._event_tokens[event_field] = [set()] * len(self.events)

        # Popularity prior from capacity (or follower_count if joined later)
        if "capacity" in self.events.columns:
            cap = pd.to_numeric(self.events["capacity"], errors="coerce").fillna(0).to_numpy(dtype=float)
            if cap.max() > 0:
                self._popularity = cap / cap.max()
            else:
                self._popularity = np.zeros(len(self.events))
        else:
            self._popularity = np.zeros(len(self.events))

        return self

    def _resolve_budget(self, user_row: pd.Series) -> Optional[float]:
        if self.budget_col and self.budget_col in user_row and pd.notna(user_row[self.budget_col]):
            try:
                return float(user_row[self.budget_col])
            except (TypeError, ValueError):
                pass
        # Activity-level proxy on global price distribution
        if self.events is None or self.price_col not in self.events.columns:
            return None
        activity = str(user_row.get("activity_level", "")).strip().lower() if "activity_level" in user_row else ""
        pct = self.ACTIVITY_BUDGET_PERCENTILE.get(activity, 50)
        prices = pd.to_numeric(self.events[self.price_col], errors="coerce").dropna()
        if prices.empty:
            return None
        return float(np.percentile(prices, pct))

    def _popularity_fallback(self, top_n: int) -> pd.DataFrame:
        """Cold-start fallback: rank by popularity prior with non-zero score."""
        events = self.events.copy()
        events["KnowledgeScore"] = self._popularity * (
            sum(w for _, _, w in self.field_pairs) + self.BUDGET_WEIGHT + self.POPULARITY_WEIGHT
        )
        return events.sort_values("KnowledgeScore", ascending=False).head(top_n)

    def recommend(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        if self.users is None or self.events is None:
            raise RuntimeError("Call fit() before recommend().")

        user_row_df = self.users[self.users[self.user_id_col] == user_id].head(1)
        if user_row_df.empty:
            return self._popularity_fallback(top_n)

        user_row = user_row_df.iloc[0]
        n_events = len(self.events)
        scores = np.zeros(n_events, dtype=float)

        # Field-based graded scoring (Jaccard-like: |∩| / max(1, |user|))
        for user_field, event_field, weight in self.field_pairs:
            if user_field not in user_row.index:
                continue
            user_tokens = _to_tokens(user_row[user_field])
            if not user_tokens:
                continue
            denom = max(1, len(user_tokens))
            event_token_lists = self._event_tokens.get(event_field, [])
            for i, ev_tokens in enumerate(event_token_lists):
                if not ev_tokens:
                    continue
                overlap = len(user_tokens & ev_tokens)
                if overlap:
                    scores[i] += weight * (overlap / denom)

        # Budget signal
        budget = self._resolve_budget(user_row)
        if budget is not None and self.price_col in self.events.columns:
            prices = pd.to_numeric(self.events[self.price_col], errors="coerce").to_numpy(dtype=float)
            within_budget = np.where(np.isnan(prices), 0.0, (prices <= budget).astype(float))
            scores += self.BUDGET_WEIGHT * within_budget

        # Popularity prior tie-breaker
        scores += self.POPULARITY_WEIGHT * self._popularity

        events = self.events.copy()
        events["KnowledgeScore"] = scores
        # Stable secondary sort on price (cheaper first when tied)
        if self.price_col in events.columns:
            return events.sort_values(["KnowledgeScore", self.price_col], ascending=[False, True]).head(top_n)
        return events.sort_values("KnowledgeScore", ascending=False).head(top_n)
