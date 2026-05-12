from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

Strategy = Literal["cold_start", "active", "trending"]


@dataclass(frozen=True)
class WeightScheme:
    alpha: float
    beta: float
    gamma: float


# Tuned via grid search on the synthetic holdout (see reports/grid_search_*.json).
# `active` weights from active-scheme grid; `cold_start` from cold_start-scheme grid.
DEFAULT_WEIGHTS: dict[Strategy, WeightScheme] = {
    "cold_start": WeightScheme(alpha=0.25, beta=0.50, gamma=0.25),
    "active": WeightScheme(alpha=0.00, beta=0.75, gamma=0.25),
    "trending": WeightScheme(alpha=0.30, beta=0.30, gamma=0.40),
}
DEFAULT_TREND_WINDOW_DAYS = 60


def _minmax(series: pd.Series) -> pd.Series:
    s = series.fillna(0.0).astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class HybridRanker:
    """Combine KnowledgeScore, GraphScore, TrendScore with strategy-based weights.

    Each component is min-max normalized to [0, 1] before the weighted sum so
    no single recommender dominates by virtue of its raw score scale.
    """

    def __init__(
        self,
        interaction_threshold: int = 5,
        weights: dict[Strategy, WeightScheme] | None = None,
    ) -> None:
        self.interaction_threshold = interaction_threshold
        self.weights = weights or DEFAULT_WEIGHTS

    def _choose_strategy(
        self,
        interactions: int,
        focus: Optional[Strategy] = None,
        activity_level: Optional[str] = None,
    ) -> Strategy:
        if focus in self.weights:
            return focus
        # Activity-level gate (step 13): explicit "low" stays cold-start
        # regardless of count; "high" jumps to active immediately.
        if activity_level:
            al = activity_level.strip().lower()
            if al == "low":
                return "cold_start"
            if al == "high":
                return "active"
        if interactions < self.interaction_threshold:
            return "cold_start"
        return "active"

    def rank(
        self,
        scores: pd.DataFrame,
        user_interactions: int,
        focus: Optional[Strategy] = None,
        top_n: int = 10,
        activity_level: Optional[str] = None,
    ) -> pd.DataFrame:
        required = ["KnowledgeScore", "GraphScore", "TrendScore"]
        missing = [c for c in required if c not in scores.columns]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")

        strategy = self._choose_strategy(user_interactions, focus, activity_level)
        scheme = self.weights[strategy]

        df = scores.copy()
        k_norm = _minmax(df["KnowledgeScore"])
        g_norm = _minmax(df["GraphScore"])
        t_norm = _minmax(df["TrendScore"])
        df["FinalScore"] = scheme.alpha * k_norm + scheme.beta * g_norm + scheme.gamma * t_norm
        return df.sort_values("FinalScore", ascending=False).head(top_n)
