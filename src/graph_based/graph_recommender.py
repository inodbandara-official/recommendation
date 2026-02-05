from __future__ import annotations

from typing import Iterable

import networkx as nx
import pandas as pd


class GraphBasedRecommender:
    """Simple graph-based recommender using item-item co-occurrence."""

    def __init__(self, user_column: str = "user_id", item_column: str = "item_id"):
        self.user_column = user_column
        self.item_column = item_column
        self.graph = nx.Graph()

    def fit(self, interactions: pd.DataFrame) -> "GraphBasedRecommender":
        """Build an undirected item co-occurrence graph."""
        if self.user_column not in interactions or self.item_column not in interactions:
            raise ValueError(f"Interactions must include '{self.user_column}' and '{self.item_column}' columns.")

        grouped = interactions.groupby(self.user_column)[self.item_column].apply(list)
        for items in grouped:
            for i, item_a in enumerate(items):
                for item_b in items[i + 1 :]:
                    if self.graph.has_edge(item_a, item_b):
                        self.graph[item_a][item_b]["weight"] += 1
                    else:
                        self.graph.add_edge(item_a, item_b, weight=1)
        return self

    def recommend(self, item_ids: Iterable[str], top_k: int = 5) -> list[str]:
        """Return items most strongly connected to the provided seed items."""
        scores: dict[str, float] = {}
        for item in item_ids:
            if item not in self.graph:
                continue
            for neighbor, attrs in self.graph[item].items():
                scores[neighbor] = scores.get(neighbor, 0.0) + attrs.get("weight", 1.0)

        ranked = [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
        return ranked[:top_k]
