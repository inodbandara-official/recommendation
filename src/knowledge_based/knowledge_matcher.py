from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


class KnowledgeMatcher:
    """Knowledge-based recommender using user profile fields (no past activity required)."""

    def __init__(
        self,
        user_id_col: str = "user_id",
        user_category_fields: Sequence[str] = ("art_interests",),
        user_location_fields: Sequence[str] = ("region_preference",),
        event_category_fields: Sequence[str] = ("art_forms", "genres"),
        event_location_fields: Sequence[str] = ("region",),
        price_col: str = "ticket_price",
        budget_col: Optional[str] = None,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ) -> None:
        self.user_id_col = user_id_col
        self.user_category_fields = tuple(user_category_fields)
        self.user_location_fields = tuple(user_location_fields)
        self.event_category_fields = tuple(event_category_fields)
        self.event_location_fields = tuple(event_location_fields)
        self.price_col = price_col
        self.budget_col = budget_col
        self.weights = weights
        self.users: Optional[pd.DataFrame] = None
        self.events: Optional[pd.DataFrame] = None

    def fit(self, users: pd.DataFrame, events: pd.DataFrame) -> "KnowledgeMatcher":
        missing_user_cols = [c for c in [self.user_id_col] if c not in users.columns]
        missing_event_cols = [c for c in [self.price_col] if c not in events.columns]
        if missing_user_cols:
            raise ValueError(f"Users data is missing required columns: {missing_user_cols}")
        if missing_event_cols:
            raise ValueError(f"Events data is missing required columns: {missing_event_cols}")

        self.users = users.copy()
        self.events = events.copy()
        return self

    def recommend(self, user_id: str, top_n: int = 10) -> pd.DataFrame:
        if self.users is None or self.events is None:
            raise RuntimeError("Call fit() before recommend().")

        user_row = self.users[self.users[self.user_id_col] == user_id].head(1)
        if user_row.empty:
            # Fallback: no profile found, return top events by lowest price
            events = self.events.copy()
            events["KnowledgeScore"] = 0.0
            return events.sort_values(self.price_col, ascending=True).head(top_n)

        user_categories = self._collect_tokens(user_row, self.user_category_fields)
        user_locations = self._collect_tokens(user_row, self.user_location_fields)
        user_budget = self._first_budget(user_row, self.budget_col) if self.budget_col else None

        cat_w, loc_w, price_w = self.weights
        events = self.events.copy()

        def score_row(row: pd.Series) -> float:
            score = 0.0

            event_categories = self._collect_tokens(row.to_frame().T, self.event_category_fields)
            if user_categories and event_categories and user_categories.intersection(event_categories):
                score += cat_w

            event_locations = self._collect_tokens(row.to_frame().T, self.event_location_fields)
            if user_locations and event_locations and user_locations.intersection(event_locations):
                score += loc_w

            price = row.get(self.price_col)
            if user_budget is not None and pd.notna(price):
                try:
                    if float(price) <= user_budget:
                        score += price_w
                except (TypeError, ValueError):
                    pass

            return score

        events["KnowledgeScore"] = events.apply(score_row, axis=1)
        return events.sort_values(["KnowledgeScore", self.price_col], ascending=[False, True]).head(top_n)

    @staticmethod
    def _collect_tokens(df: pd.DataFrame, columns: Iterable[str]) -> set[str]:
        tokens: set[str] = set()
        for col in columns:
            if col not in df.columns:
                continue
            val = df.iloc[0][col]
            tokens.update(KnowledgeMatcher._to_tokens(val))
        return tokens

    @staticmethod
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

    @staticmethod
    def _first_budget(df: pd.DataFrame, budget_col: str) -> Optional[float]:
        if budget_col in df.columns:
            val = df.iloc[0][budget_col]
            if pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
        return None
