"""Offline evaluation harness with optional grid search.

Temporal holdout on `attends`: hold the most recent `holdout_frac` of each
user's attendance as test, keep earlier as train. For each test user, run the
hybrid recommender on the train slice and score the top-K against the held-out
events using precision/recall/MAP/NDCG/coverage/diversity.

Usage:
    python pipeline/8_evaluate.py                    # baseline
    python pipeline/8_evaluate.py --grid             # grid search weights+window
    python pipeline/8_evaluate.py --graph-mode pagerank
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_dataset  # noqa: E402
from src.knowledge_based import KnowledgeMatcher  # noqa: E402
from src.graph_based import recommend_from_similar_users, HeteroGraphRecommender  # noqa: E402
from src.trend_based import TrendWindowRecommender  # noqa: E402
from src.hybrid.hybrid_ranker import HybridRanker, WeightScheme  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


# ─── Data split ──────────────────────────────────────────────────────────────

def temporal_split(
    attends: pd.DataFrame,
    holdout_frac: float = 0.2,
    min_history: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Hold out the latest `holdout_frac` of each user's attends as test.

    Users with fewer than `min_history` total attends are kept entirely in
    train and excluded from the test set.
    """
    df = attends.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])

    train_rows: List[pd.DataFrame] = []
    test_truth: Dict[str, Set[str]] = {}
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        if n < min_history:
            train_rows.append(group)
            continue
        n_test = max(1, int(round(n * holdout_frac)))
        n_test = min(n_test, n - 1)  # always keep at least 1 train row
        train_part = group.iloc[: n - n_test]
        test_part = group.iloc[n - n_test:]
        train_rows.append(train_part)
        test_truth[str(user_id)] = set(test_part["event_id"].astype(str))

    train = pd.concat(train_rows, ignore_index=True) if train_rows else df.iloc[0:0]
    return train, test_truth


# ─── Per-user scoring (avoids re-fitting on every call) ──────────────────────

def _score_user(
    user_id: str,
    users: pd.DataFrame,
    events: pd.DataFrame,
    attends_train: pd.DataFrame,
    follows: pd.DataFrame,
    km: KnowledgeMatcher,
    trend_df: pd.DataFrame,
    graph_mode: str,
    pagerank_rec: Optional[HeteroGraphRecommender],
    ranker: HybridRanker,
    top_n: int,
    candidate_ids: Set[str],
) -> List[str]:
    # Knowledge
    k_df = km.recommend(user_id, top_n=len(events))[["event_id", "KnowledgeScore"]]

    # Graph
    if graph_mode == "pagerank" and pagerank_rec is not None:
        pairs = pagerank_rec.recommend_events_for_user(user_id, top_k=top_n * 5)
        g_df = pd.DataFrame(
            [(p[0][len("event:"):], p[1]) for p in pairs],
            columns=["event_id", "GraphScore"],
        )
    else:
        g_df = recommend_from_similar_users(
            attends=attends_train,
            follows=follows,
            target_user=user_id,
            top_users=50,
            top_n=top_n * 5,
        )

    # Restrict to candidate set
    k_df = k_df[k_df["event_id"].astype(str).isin(candidate_ids)]
    g_df = g_df[g_df["event_id"].astype(str).isin(candidate_ids)]
    t_df = trend_df[trend_df["event_id"].astype(str).isin(candidate_ids)]

    # Merge & rank
    merged = (
        pd.DataFrame({"event_id": list(set(k_df["event_id"]) | set(g_df["event_id"]) | set(t_df["event_id"]))})
        .merge(k_df, on="event_id", how="left")
        .merge(g_df, on="event_id", how="left")
        .merge(t_df[["event_id", "TrendScore"]], on="event_id", how="left")
    )
    if merged.empty:
        return []
    merged[["KnowledgeScore", "GraphScore", "TrendScore"]] = merged[
        ["KnowledgeScore", "GraphScore", "TrendScore"]
    ].fillna(0.0)

    # Exclude events the user already attended in the train slice
    seen = set(attends_train.loc[attends_train["user_id"] == user_id, "event_id"].astype(str))
    if seen:
        merged = merged[~merged["event_id"].astype(str).isin(seen)]
    if merged.empty:
        return []

    user_row = users.loc[users["user_id"] == user_id].head(1)
    activity = (
        str(user_row.iloc[0]["activity_level"])
        if not user_row.empty and "activity_level" in user_row.columns
        else None
    )
    ranked = ranker.rank(
        merged,
        user_interactions=int((attends_train["user_id"] == user_id).sum()),
        top_n=top_n,
        activity_level=activity,
    )
    return ranked["event_id"].astype(str).tolist()


def _build_event_features(events: pd.DataFrame) -> Dict[str, Set[str]]:
    feats: Dict[str, Set[str]] = {}
    for _, row in events.iterrows():
        eid = str(row.get("event_id"))
        toks: Set[str] = set()
        for col in ("art_forms", "genres", "moods"):
            v = row.get(col)
            if isinstance(v, (list, tuple, set)):
                toks.update(str(x).strip().lower() for x in v if x is not None)
            elif isinstance(v, str):
                s = v.strip().strip("[]")
                toks.update(p.strip().strip("'\" ").lower() for p in s.split(",") if p.strip())
        feats[eid] = toks
    return feats


# ─── Run evaluation for one config ───────────────────────────────────────────

def run_eval(
    users: pd.DataFrame,
    events: pd.DataFrame,
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    holdout_frac: float = 0.2,
    top_k: int = 10,
    graph_mode: str = "similarity",
    trend_window_days: int = 14,
    weights: Optional[Dict[str, WeightScheme]] = None,
    interaction_threshold: int = 5,
    max_users: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, float]:
    train, truth = temporal_split(attends, holdout_frac=holdout_frac)
    users_to_eval = list(truth.keys())
    if max_users and len(users_to_eval) > max_users:
        rng = np.random.default_rng(seed)
        users_to_eval = list(rng.choice(users_to_eval, size=max_users, replace=False))

    km = KnowledgeMatcher().fit(users, events)

    if not train.empty:
        trend_df = TrendWindowRecommender().fit(train).recommend(top_n=len(events), window_days=trend_window_days)
        if trend_df.empty:
            trend_df = pd.DataFrame(columns=["event_id", "TrendScore"])
    else:
        trend_df = pd.DataFrame(columns=["event_id", "TrendScore"])

    pagerank_rec: Optional[HeteroGraphRecommender] = None
    if graph_mode == "pagerank":
        pagerank_rec = HeteroGraphRecommender()
        pagerank_rec.build_from_frames(attends=train, follows=follows, events=None)

    ranker = HybridRanker(interaction_threshold=interaction_threshold, weights=weights)
    candidate_ids = set(events["event_id"].astype(str))
    rec_map: Dict[str, List[str]] = {}
    for u in users_to_eval:
        rec_map[u] = _score_user(
            user_id=u,
            users=users,
            events=events,
            attends_train=train,
            follows=follows,
            km=km,
            trend_df=trend_df,
            graph_mode=graph_mode,
            pagerank_rec=pagerank_rec,
            ranker=ranker,
            top_n=top_k,
            candidate_ids=candidate_ids,
        )

    item_features = _build_event_features(events)
    metrics = evaluate(
        rec_map=rec_map,
        rel_map=truth,
        k=top_k,
        catalog=candidate_ids,
        item_features=item_features,
    )
    metrics["users_evaluated"] = len(users_to_eval)
    return metrics


# ─── Grid search ─────────────────────────────────────────────────────────────

def grid_search(
    users: pd.DataFrame,
    events: pd.DataFrame,
    attends: pd.DataFrame,
    follows: pd.DataFrame,
    top_k: int,
    graph_mode: str,
    holdout_frac: float,
    max_users: Optional[int],
    scheme: str = "active",
    fixed_active: WeightScheme = WeightScheme(0.0, 0.75, 0.25),
    fixed_cold_start: WeightScheme = WeightScheme(0.5, 0.2, 0.3),
) -> Tuple[List[dict], dict]:
    """Search over (alpha, beta, gamma) for one strategy and trend window.

    `scheme` selects which weight bucket to tune ("active" or "cold_start").
    The other bucket is held at its `fixed_*` default. Picks the best by NDCG@K.
    """
    step = 0.25
    grid_simplex = [
        (round(a, 4), round(b, 4), round(1.0 - a - b, 4))
        for a in np.arange(0.0, 1.01, step)
        for b in np.arange(0.0, 1.01 - a, step)
        if 1.0 - a - b >= -1e-9
    ]
    windows = [7, 14, 30, 60]

    results: List[dict] = []
    best = {"ndcg": -1.0}
    for window in windows:
        for w in grid_simplex:
            tuned = WeightScheme(*w)
            if scheme == "cold_start":
                weights = {"cold_start": tuned, "active": fixed_active, "trending": WeightScheme(0.3, 0.3, 0.4)}
            else:
                weights = {"cold_start": fixed_cold_start, "active": tuned, "trending": WeightScheme(0.3, 0.3, 0.4)}
            m = run_eval(
                users=users,
                events=events,
                attends=attends,
                follows=follows,
                holdout_frac=holdout_frac,
                top_k=top_k,
                graph_mode=graph_mode,
                trend_window_days=window,
                weights=weights,
                max_users=max_users,
            )
            entry = {
                "window": int(window),
                "scheme": scheme,
                "weights": list(w),
                "ndcg": m["ndcg"],
                "recall@k": m["recall@k"],
                "precision@k": m["precision@k"],
                "map": m["map"],
                "coverage": m["coverage"],
            }
            results.append(entry)
            if m["ndcg"] > best["ndcg"]:
                best = {**entry, "metrics": m}
    return results, best


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--graph-mode", choices=["similarity", "pagerank"], default="similarity")
    ap.add_argument("--trend-window", type=int, default=14)
    ap.add_argument("--max-users", type=int, default=500, help="cap test users for speed")
    ap.add_argument("--grid", action="store_true", help="run grid search over weights+window")
    ap.add_argument("--scheme", choices=["active", "cold_start"], default="active",
                    help="which weight bucket to tune in grid search")
    ap.add_argument("--out", default=str(REPORTS_DIR / "metrics.json"))
    args = ap.parse_args()

    REPORTS_DIR.mkdir(exist_ok=True)
    ds = load_dataset(DATA_DIR)
    users, events, attends, follows = ds["users"], ds["events"], ds["attends"], ds["follows"]
    print(f"Dataset: users={len(users):,} events={len(events):,} attends={len(attends):,} follows={len(follows):,}")

    if args.grid:
        print("Running grid search (this is the slow part)...")
        t0 = time.time()
        results, best = grid_search(
            users, events, attends, follows,
            top_k=args.top_k,
            graph_mode=args.graph_mode,
            holdout_frac=args.holdout_frac,
            max_users=args.max_users,
            scheme=args.scheme,
        )
        elapsed = time.time() - t0
        out = {
            "mode": "grid",
            "scheme_tuned": args.scheme,
            "graph_mode": args.graph_mode,
            "elapsed_seconds": elapsed,
            "n_configs": len(results),
            "best": best,
            "all_results": results,
        }
        out_path = Path(args.out).with_name(f"grid_search_{args.scheme}.json")
    else:
        print("Running baseline evaluation...")
        t0 = time.time()
        metrics = run_eval(
            users, events, attends, follows,
            holdout_frac=args.holdout_frac,
            top_k=args.top_k,
            graph_mode=args.graph_mode,
            trend_window_days=args.trend_window,
            max_users=args.max_users,
        )
        elapsed = time.time() - t0
        out = {
            "mode": "baseline",
            "graph_mode": args.graph_mode,
            "trend_window": args.trend_window,
            "top_k": args.top_k,
            "holdout_frac": args.holdout_frac,
            "max_users": args.max_users,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }
        out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "all_results"}, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
