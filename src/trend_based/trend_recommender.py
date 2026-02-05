from __future__ import annotations

from typing import Optional

import pandas as pd


class TrendBasedRecommender:
    """Popularity-based recommender using recent interaction counts."""

    def __init__(self, item_column: str = "item_id", timestamp_column: str | None = None):
        self.item_column = item_column
        self.timestamp_column = timestamp_column
        self._scores: pd.Series | None = None

    def fit(self, interactions: pd.DataFrame, decay: float | None = None) -> "TrendBasedRecommender":
        """Compute item popularity, optionally applying exponential decay over time."""
        if self.item_column not in interactions:
            raise ValueError(f"Interactions must include '{self.item_column}' column.")

        if decay is not None and self.timestamp_column:
            if self.timestamp_column not in interactions:
                raise ValueError(f"Interactions must include '{self.timestamp_column}' column when decay is provided.")
            interactions = interactions.sort_values(self.timestamp_column)
            weights = pd.Series((1 - decay) ** (len(interactions) - 1 - interactions.index.to_series()), index=interactions.index)
            self._scores = (weights.groupby(interactions[self.item_column]).sum()).sort_values(ascending=False)
        else:
            self._scores = interactions[self.item_column].value_counts()
        return self

    def recommend(self, top_k: int = 5) -> list[str]:
        """Return the top trending items."""
        if self._scores is None:
            raise RuntimeError("Call fit() before recommend().")
        return self._scores.head(top_k).index.tolist()


class TrendWindowRecommender:
    """Trending events using recent-window attendance and growth vs prior window."""

    def __init__(self, event_column: str = "event_id", timestamp_column: str = "timestamp"):
        self.event_column = event_column
        self.timestamp_column = timestamp_column
        self._attends: Optional[pd.DataFrame] = None

    def fit(self, attends: pd.DataFrame) -> "TrendWindowRecommender":
        if self.event_column not in attends or self.timestamp_column not in attends:
            raise ValueError(
                f"Attends data must include '{self.event_column}' and '{self.timestamp_column}' columns."
            )
        df = attends.copy()
        df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column], errors="coerce")
        df = df.dropna(subset=[self.timestamp_column])
        self._attends = df
        return self

    def recommend(
        self,
        top_n: int = 10,
        window_days: int = 14,
        prev_window_days: Optional[int] = None,
        now: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if self._attends is None:
            raise RuntimeError("Call fit() before recommend().")

        prev_window_days = prev_window_days or window_days
        df = self._attends
        ts_col = self.timestamp_column

        now_ts = now or df[ts_col].max()
        if pd.isna(now_ts):
            now_ts = pd.Timestamp.utcnow()

        recent_start = now_ts - pd.Timedelta(days=window_days)
        prev_start = recent_start - pd.Timedelta(days=prev_window_days)

        recent_mask = (df[ts_col] >= recent_start) & (df[ts_col] <= now_ts)
        prev_mask = (df[ts_col] >= prev_start) & (df[ts_col] < recent_start)

        recent_counts = df.loc[recent_mask].groupby(self.event_column).size()
        prev_counts = df.loc[prev_mask].groupby(self.event_column).size()

        all_events = set(recent_counts.index) | set(prev_counts.index)
        if not all_events:
            return pd.DataFrame(columns=["event_id", "recent_count", "prev_count", "growth_rate", "TrendScore"])

        recent = pd.Series({e: recent_counts.get(e, 0) for e in all_events})
        prev = pd.Series({e: prev_counts.get(e, 0) for e in all_events})

        growth_rate = (recent - prev) / (prev.replace(0, 1))
        raw_score = recent + growth_rate

        min_score = raw_score.min()
        max_score = raw_score.max()
        if pd.isna(min_score) or pd.isna(max_score):
            return pd.DataFrame(columns=["event_id", "recent_count", "prev_count", "growth_rate", "TrendScore"])

        if max_score == min_score:
            trend_score = pd.Series(1.0, index=raw_score.index)
        else:
            trend_score = (raw_score - min_score) / (max_score - min_score)

        result = pd.DataFrame(
            {
                "event_id": list(all_events),
                "recent_count": recent.loc[list(all_events)].values,
                "prev_count": prev.loc[list(all_events)].values,
                "growth_rate": growth_rate.loc[list(all_events)].values,
                "TrendScore": trend_score.loc[list(all_events)].values,
            }
        )

        return result.sort_values(["TrendScore", "recent_count"], ascending=[False, False]).head(top_n)
