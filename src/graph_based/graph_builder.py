from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import pandas as pd


class HeteroGraphRecommender:
    """Graph-based recommender over users, events, and artists using personalized PageRank."""

    def __init__(
        self,
        user_prefix: str = "user:",
        event_prefix: str = "event:",
        artist_prefix: str = "artist:",
        alpha: float = 0.85,
    ) -> None:
        self.user_prefix = user_prefix
        self.event_prefix = event_prefix
        self.artist_prefix = artist_prefix
        self.alpha = alpha
        self.graph = nx.DiGraph()

    def _user_node(self, user_id: str) -> str:
        return f"{self.user_prefix}{user_id}"

    def _event_node(self, event_id: str) -> str:
        return f"{self.event_prefix}{event_id}"

    def _artist_node(self, artist_id: str) -> str:
        return f"{self.artist_prefix}{artist_id}"

    def build_from_frames(
        self,
        attends: Optional[pd.DataFrame] = None,
        follows: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> "HeteroGraphRecommender":
        g = nx.DiGraph()

        if attends is not None:
            for _, row in attends.iterrows():
                user_id = row.get("user_id")
                event_id = row.get("event_id")
                if pd.isna(user_id) or pd.isna(event_id):
                    continue
                u = self._user_node(str(user_id))
                e = self._event_node(str(event_id))
                g.add_edge(u, e, relation="attended")
                g.add_edge(e, u, relation="attended_rev")

        if follows is not None:
            for _, row in follows.iterrows():
                user_id = row.get("user_id")
                artist_id = row.get("artist_id")
                if pd.isna(user_id) or pd.isna(artist_id):
                    continue
                u = self._user_node(str(user_id))
                a = self._artist_node(str(artist_id))
                g.add_edge(u, a, relation="followed")
                g.add_edge(a, u, relation="followed_rev")

        if events is not None:
            for _, row in events.iterrows():
                event_id = row.get("event_id")
                artist_id = row.get("artist_id")
                if pd.isna(event_id):
                    continue
                e = self._event_node(str(event_id))
                g.add_node(e, kind="event")
                if not pd.isna(artist_id):
                    a = self._artist_node(str(artist_id))
                    g.add_edge(e, a, relation="performed_by")
                    g.add_edge(a, e, relation="performed_by_rev")

        self.graph = g
        return self

    def recommend_events_for_user(self, user_id: str, top_k: int = 10, exclude_attended: bool = True) -> list[tuple[str, float]]:
        if not self.graph:
            raise RuntimeError("Graph is empty. Build the graph before recommending.")

        user_node = self._user_node(str(user_id))
        if user_node not in self.graph:
            return []

        personalization = {node: 0.0 for node in self.graph.nodes}
        personalization[user_node] = 1.0

        scores = nx.pagerank(self.graph, alpha=self.alpha, personalization=personalization)

        attended = set()
        if exclude_attended:
            for nbr in self.graph.successors(user_node):
                if nbr.startswith(self.event_prefix):
                    attended.add(nbr)

        event_scores = [
            (node, score)
            for node, score in scores.items()
            if node.startswith(self.event_prefix) and node not in attended
        ]

        event_scores.sort(key=lambda kv: kv[1], reverse=True)
        return event_scores[:top_k]


def load_graph_from_csvs(
    data_dir: Path = Path("data"),
    attends_file: str = "attends.csv",
    follows_file: str = "follows.csv",
    events_file: str = "events.csv",
) -> HeteroGraphRecommender:
    """Utility to load CSVs and build the heterogeneous graph."""
    attends_df = None
    follows_df = None
    events_df = None

    attends_path = data_dir / attends_file
    follows_path = data_dir / follows_file
    events_path = data_dir / events_file

    if attends_path.exists():
        attends_df = pd.read_csv(attends_path)
    if follows_path.exists():
        follows_df = pd.read_csv(follows_path)
    if events_path.exists():
        events_df = pd.read_csv(events_path)

    recommender = HeteroGraphRecommender()
    recommender.build_from_frames(attends=attends_df, follows=follows_df, events=events_df)
    return recommender
