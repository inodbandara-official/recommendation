from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import networkx as nx
import pandas as pd


def _build_user_item_sets(attends: pd.DataFrame, follows: pd.DataFrame) -> Dict[str, set]:
    """Combine attended events and followed artists per user for Jaccard."""
    user_items: Dict[str, set] = {}

    if attends is not None:
        for _, row in attends.iterrows():
            u = row.get("user_id")
            e = row.get("event_id")
            if pd.isna(u) or pd.isna(e):
                continue
            user_items.setdefault(str(u), set()).add(f"event:{e}")

    if follows is not None:
        for _, row in follows.iterrows():
            u = row.get("user_id")
            a = row.get("artist_id")
            if pd.isna(u) or pd.isna(a):
                continue
            user_items.setdefault(str(u), set()).add(f"artist:{a}")

    return user_items


def jaccard_similar_users(attends: pd.DataFrame, follows: pd.DataFrame, target_user: str) -> Dict[str, float]:
    user_items = _build_user_item_sets(attends, follows)
    if target_user not in user_items:
        return {}
    target_set = user_items[target_user]
    sims: Dict[str, float] = {}
    for user, items in user_items.items():
        if user == target_user:
            continue
        union = target_set | items
        if not union:
            continue
        sims[user] = len(target_set & items) / len(union)
    return sims


def _build_bipartite_graph(attends: pd.DataFrame) -> nx.Graph:
    """Build user-event bipartite graph for Adamic-Adar on user projection."""
    B = nx.Graph()
    if attends is None:
        return B
    for _, row in attends.iterrows():
        u = row.get("user_id")
        e = row.get("event_id")
        if pd.isna(u) or pd.isna(e):
            continue
        user_node = f"user:{u}"
        event_node = f"event:{e}"
        B.add_node(user_node, bipartite="user")
        B.add_node(event_node, bipartite="event")
        B.add_edge(user_node, event_node)
    return B


def adamic_adar_similar_users(attends: pd.DataFrame, target_user: str) -> Dict[str, float]:
    B = _build_bipartite_graph(attends)
    target_node = f"user:{target_user}"
    if target_node not in B:
        return {}
    # Project to user-user graph via shared events
    users = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "user"]
    projected = nx.algorithms.bipartite.weighted_projected_graph(B, users)
    sims: Dict[str, float] = {}
    if target_node not in projected:
        return sims
    # Adamic-Adar on user projection
    for u, v, score in nx.adamic_adar_index(projected, ebunch=((target_node, n) for n in projected if n != target_node)):
        sims[v.replace("user:", "")] = float(score)
    return sims


def merge_similarity(
    jaccard_scores: Dict[str, float],
    aa_scores: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    """Combine Jaccard and Adamic-Adar with weight alpha for Jaccard."""
    users = set(jaccard_scores) | set(aa_scores)
    merged: Dict[str, float] = {}
    for u in users:
        j = jaccard_scores.get(u, 0.0)
        a = aa_scores.get(u, 0.0)
        merged[u] = alpha * j + (1 - alpha) * a
    return merged


def recommend_from_similar_users(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    target_user: str,
    top_users: int = 20,
    top_n: int = 10,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Recommend events based on similar users' attendance.

    - Compute Jaccard over attended events + followed artists.
    - Compute Adamic-Adar over user projection of the attend bipartite graph.
    - Merge similarities and use them to score candidate events not yet attended by the target user.
    """
    j_scores = jaccard_similar_users(attends, follows, target_user)
    aa_scores = adamic_adar_similar_users(attends, target_user)
    merged = merge_similarity(j_scores, aa_scores, alpha=alpha)

    if not merged:
        return pd.DataFrame(columns=["event_id", "GraphScore"])

    sorted_users = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:top_users]
    sim_users = [u for u, _ in sorted_users]
    sim_map = dict(sorted_users)

    target_attended = set()
    if attends is not None:
        target_attended = set(attends.loc[attends["user_id"] == target_user, "event_id"].astype(str))

    scores: Dict[str, float] = {}
    if attends is not None:
        for _, row in attends.iterrows():
            u = row.get("user_id")
            e = row.get("event_id")
            if pd.isna(u) or pd.isna(e):
                continue
            u_str = str(u)
            e_str = str(e)
            if u_str not in sim_map:
                continue
            if e_str in target_attended:
                continue
            scores[e_str] = scores.get(e_str, 0.0) + sim_map[u_str]

    if not scores:
        return pd.DataFrame(columns=["event_id", "GraphScore"])

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return pd.DataFrame(ranked, columns=["event_id", "GraphScore"])
