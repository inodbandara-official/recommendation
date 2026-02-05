from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd


class KnowledgeBasedRecommender:
    """Rule-driven recommender that applies pre-defined constraints."""

    def __init__(self, rules: Mapping[str, Iterable[str]] | None = None, user_column: str = "user_id", item_column: str = "item_id"):
        self.rules = {k: list(v) for k, v in (rules or {}).items()}
        self.user_column = user_column
        self.item_column = item_column
        self._interactions: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame) -> "KnowledgeBasedRecommender":
        """Store interactions; apply rule normalization if needed."""
        if self.user_column not in interactions or self.item_column not in interactions:
            raise ValueError(f"Interactions must include '{self.user_column}' and '{self.item_column}' columns.")
        self._interactions = interactions.copy()
        return self

    def recommend(self, user_id: str, top_k: int = 5) -> list[str]:
        """Return items matching any configured rules for the user."""
        if self._interactions is None:
            raise RuntimeError("Call fit() before recommend().")

        allowed_items = set()
        for rule_key, allowed in self.rules.items():
            if rule_key == "default" or rule_key == str(user_id):
                allowed_items.update(allowed)

        # If no rule matches, fall back to items the user has not interacted with.
        user_items = set(self._interactions[self._interactions[self.user_column] == user_id][self.item_column])
        candidate_items = allowed_items or set(self._interactions[self.item_column].unique())
        ranked = [item for item in candidate_items if item not in user_items]
        return ranked[:top_k]
