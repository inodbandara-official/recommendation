from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

Strategy = Literal["cold_start", "active", "trending"]


@dataclass(frozen=True)
class WeightScheme:
    alpha: float
    beta: float
    gamma: float


DEFAULT_WEIGHTS: dict[Strategy, WeightScheme] = {
    "cold_start": WeightScheme(alpha=0.5, beta=0.2, gamma=0.3),
    "active": WeightScheme(alpha=0.2, beta=0.5, gamma=0.3),
    "trending": WeightScheme(alpha=0.3, beta=0.3, gamma=0.4),
}


class HybridRanker:
    """Combine KnowledgeScore, GraphScore, TrendScore with strategy-based weights."""

    def __init__(
        self,
        interaction_threshold: int = 5,
        weights: dict[Strategy, WeightScheme] | None = None,
    ) -> None:
        self.interaction_threshold = interaction_threshold
        self.weights = weights or DEFAULT_WEIGHTS

    def _choose_strategy(self, interactions: int, focus: Optional[Strategy] = None) -> Strategy:
        if focus in self.weights:
            return focus
        if interactions < self.interaction_threshold:
            return "cold_start"
        return "active"

    def rank(
        self,
        scores: pd.DataFrame,
        user_interactions: int,
        focus: Optional[Strategy] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Rank events by weighted combination of score columns.

        Expected columns: KnowledgeScore, GraphScore, TrendScore.
        """
        required = ["KnowledgeScore", "GraphScore", "TrendScore"]
        missing = [c for c in required if c not in scores.columns]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")

        strategy = self._choose_strategy(user_interactions, focus)
        scheme = self.weights[strategy]

        df = scores.copy()
        df["FinalScore"] = (
            scheme.alpha * df["KnowledgeScore"].fillna(0)
            + scheme.beta * df["GraphScore"].fillna(0)
            + scheme.gamma * df["TrendScore"].fillna(0)
        )
        return df.sort_values("FinalScore", ascending=False).head(top_n)
