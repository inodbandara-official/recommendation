from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


class HybridRecommender:
    """Combine ranked lists from multiple recommenders using weighted voting."""

    def __init__(self, weights: Sequence[float] | None = None):
        self.weights = weights

    def blend(self, recommendations: Sequence[Iterable[str]], top_k: int = 5) -> list[str]:
        """Blend multiple recommendation lists into a single ranking."""
        counter: Counter[str] = Counter()
        for idx, recs in enumerate(recommendations):
            weight = 1.0
            if self.weights and idx < len(self.weights):
                weight = self.weights[idx]
            for rank, item in enumerate(recs):
                counter[item] += weight / (rank + 1)

        ranked = [item for item, _ in counter.most_common(top_k)]
        return ranked
