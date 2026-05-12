"""Vectorized user-similarity and graph recommendations.

Replaces the previous O(U^2) Python-level implementation with sparse-matrix
operations sized for ~10k users / ~150k interactions. Public function names
are preserved for backwards compatibility.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _build_user_item_matrix(
    attends: Optional[pd.DataFrame],
    follows: Optional[pd.DataFrame],
) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
    """Build a sparse user x (event ∪ artist) binary matrix.

    Returns the matrix plus two index maps: user -> row, item -> col.
    Items are namespaced ('event:E0001', 'artist:A0001') so events and
    artists never collide.
    """
    user_idx: Dict[str, int] = {}
    item_idx: Dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []

    def _add(u: str, item_key: str) -> None:
        if u not in user_idx:
            user_idx[u] = len(user_idx)
        if item_key not in item_idx:
            item_idx[item_key] = len(item_idx)
        rows.append(user_idx[u])
        cols.append(item_idx[item_key])

    if attends is not None and not attends.empty:
        u_arr = attends["user_id"].astype(str).to_numpy()
        e_arr = attends["event_id"].astype(str).to_numpy()
        for u, e in zip(u_arr, e_arr):
            if u and e and u != "nan" and e != "nan":
                _add(u, f"event:{e}")

    if follows is not None and not follows.empty:
        u_arr = follows["user_id"].astype(str).to_numpy()
        a_arr = follows["artist_id"].astype(str).to_numpy()
        for u, a in zip(u_arr, a_arr):
            if u and a and u != "nan" and a != "nan":
                _add(u, f"artist:{a}")

    n_users = max(1, len(user_idx))
    n_items = max(1, len(item_idx))
    data = np.ones(len(rows), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    # Deduplicate (binary occurrence)
    mat.data = np.ones_like(mat.data)
    mat.sum_duplicates()
    mat.data = np.minimum(mat.data, 1.0)
    return mat, user_idx, item_idx


def _similar_users_vector(
    matrix: csr_matrix,
    user_idx: Dict[str, int],
    target_user: str,
) -> Optional[np.ndarray]:
    """Return a (n_users,) cosine-similarity vector for the target user."""
    if target_user not in user_idx:
        return None
    row = matrix[user_idx[target_user]]
    if row.nnz == 0:
        return None
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    target_norm = norms[user_idx[target_user]]
    if target_norm == 0:
        return None
    dot = matrix.dot(row.T).toarray().ravel()
    safe_norms = np.where(norms == 0, 1.0, norms)
    sims = dot / (safe_norms * target_norm)
    sims[user_idx[target_user]] = 0.0
    return sims


def jaccard_similar_users(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    target_user: str,
) -> Dict[str, float]:
    """Cosine similarity over (events ∪ artists). Name kept for back-compat."""
    matrix, user_idx, _ = _build_user_item_matrix(attends, follows)
    sims = _similar_users_vector(matrix, user_idx, target_user)
    if sims is None:
        return {}
    inv = {i: u for u, i in user_idx.items()}
    return {inv[i]: float(sims[i]) for i in np.where(sims > 0)[0]}


def adamic_adar_similar_users(
    attends: pd.DataFrame,
    target_user: str,
) -> Dict[str, float]:
    """Adamic-Adar style score over the user-event bipartite graph.

    Vectorized as M @ diag(1/log(deg+1)) @ M.T applied to target row.
    """
    matrix, user_idx, _ = _build_user_item_matrix(attends, None)
    if target_user not in user_idx:
        return {}
    item_degree = np.asarray(matrix.sum(axis=0)).ravel()
    weights = np.zeros_like(item_degree, dtype=np.float32)
    mask = item_degree > 1
    weights[mask] = 1.0 / np.log(item_degree[mask] + 1.0)
    weighted = matrix.multiply(weights)
    target_row = matrix[user_idx[target_user]]
    scores = weighted.dot(target_row.T).toarray().ravel()
    scores[user_idx[target_user]] = 0.0
    inv = {i: u for u, i in user_idx.items()}
    return {inv[i]: float(scores[i]) for i in np.where(scores > 0)[0]}


def merge_similarity(
    jaccard_scores: Dict[str, float],
    aa_scores: Dict[str, float],
    alpha: float = 0.5,
) -> Dict[str, float]:
    users = set(jaccard_scores) | set(aa_scores)
    return {u: alpha * jaccard_scores.get(u, 0.0) + (1 - alpha) * aa_scores.get(u, 0.0) for u in users}


def recommend_from_similar_users(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    target_user: str,
    top_users: int = 50,
    top_n: int = 10,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Score events by attendance of top-K similar users (vectorized)."""
    matrix, user_idx, item_idx = _build_user_item_matrix(attends, follows)
    sims = _similar_users_vector(matrix, user_idx, target_user)
    if sims is None:
        return pd.DataFrame(columns=["event_id", "GraphScore"])

    if top_users < len(sims):
        cutoff_idx = np.argpartition(-sims, top_users)[:top_users]
        sim_mask = np.zeros_like(sims)
        sim_mask[cutoff_idx] = sims[cutoff_idx]
    else:
        sim_mask = sims

    # Score events: similar_users x event_subset
    event_cols = {k: v for k, v in item_idx.items() if k.startswith("event:")}
    if not event_cols:
        return pd.DataFrame(columns=["event_id", "GraphScore"])

    sim_row = csr_matrix(sim_mask.reshape(1, -1))
    event_scores = sim_row.dot(matrix).toarray().ravel()

    # Mask out events the target user already attended
    target_row = matrix[user_idx[target_user]].toarray().ravel()
    event_scores = event_scores * (1.0 - target_row)

    # Restrict to event columns
    inv_item = {i: k for k, i in item_idx.items()}
    rows = []
    for col_idx, score in enumerate(event_scores):
        if score <= 0:
            continue
        key = inv_item.get(col_idx, "")
        if key.startswith("event:"):
            rows.append((key[len("event:"):], float(score)))
    if not rows:
        return pd.DataFrame(columns=["event_id", "GraphScore"])

    df = pd.DataFrame(rows, columns=["event_id", "GraphScore"])
    return df.sort_values("GraphScore", ascending=False).head(top_n).reset_index(drop=True)


def recommend_artists_from_similar_users(
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    target_user: str,
    top_users: int = 50,
    top_n: int = 10,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Score artists by follow patterns of top-K similar users (vectorized)."""
    matrix, user_idx, item_idx = _build_user_item_matrix(attends, follows)
    sims = _similar_users_vector(matrix, user_idx, target_user)
    if sims is None:
        return pd.DataFrame(columns=["artist_id", "ArtistGraphScore"])

    if top_users < len(sims):
        cutoff_idx = np.argpartition(-sims, top_users)[:top_users]
        sim_mask = np.zeros_like(sims)
        sim_mask[cutoff_idx] = sims[cutoff_idx]
    else:
        sim_mask = sims

    sim_row = csr_matrix(sim_mask.reshape(1, -1))
    artist_scores = sim_row.dot(matrix).toarray().ravel()
    target_row = matrix[user_idx[target_user]].toarray().ravel()
    artist_scores = artist_scores * (1.0 - target_row)

    inv_item = {i: k for k, i in item_idx.items()}
    rows = []
    for col_idx, score in enumerate(artist_scores):
        if score <= 0:
            continue
        key = inv_item.get(col_idx, "")
        if key.startswith("artist:"):
            rows.append((key[len("artist:"):], float(score)))
    if not rows:
        return pd.DataFrame(columns=["artist_id", "ArtistGraphScore"])

    df = pd.DataFrame(rows, columns=["artist_id", "ArtistGraphScore"])
    return df.sort_values("ArtistGraphScore", ascending=False).head(top_n).reset_index(drop=True)
