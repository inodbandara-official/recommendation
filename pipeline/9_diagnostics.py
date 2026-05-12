"""Diagnostics for the recommender.

Currently runs:
  - Knowledge / Graph / Trend correlation with held-out attendance.

This answers "is the synthetic dataset's stated user profile actually
predictive of attendance?" — the question raised when grid search picked
alpha=0 (no Knowledge weight) for active users.

Usage:
    python pipeline/9_diagnostics.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from src.data_io import load_dataset  # noqa: E402
from src.knowledge_based import KnowledgeMatcher  # noqa: E402
from src.graph_based import recommend_from_similar_users  # noqa: E402
from src.trend_based import TrendWindowRecommender  # noqa: E402

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def temporal_split(attends: pd.DataFrame, holdout_frac: float = 0.2):
    df = attends.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])
    train_rows, truth = [], {}
    for user_id, group in df.groupby("user_id"):
        n = len(group)
        if n < 2:
            train_rows.append(group)
            continue
        n_test = max(1, int(round(n * holdout_frac)))
        n_test = min(n_test, n - 1)
        train_rows.append(group.iloc[: n - n_test])
        truth[str(user_id)] = set(group.iloc[n - n_test:]["event_id"].astype(str))
    train = pd.concat(train_rows, ignore_index=True) if train_rows else df.iloc[0:0]
    return train, truth


def correlation_report(max_users: int = 200, seed: int = 42):
    ds = load_dataset(DATA_DIR)
    users, events, attends, follows = ds["users"], ds["events"], ds["attends"], ds["follows"]

    train, truth = temporal_split(attends)
    eligible = list(truth.keys())
    rng = np.random.default_rng(seed)
    if len(eligible) > max_users:
        eligible = list(rng.choice(eligible, size=max_users, replace=False))

    print(f"Diagnostic: correlating component scores with held-out attendance")
    print(f"  users sampled: {len(eligible)}")
    print(f"  events catalog: {len(events):,}")

    km = KnowledgeMatcher().fit(users, events)
    trend_df = TrendWindowRecommender().fit(train).recommend(top_n=len(events), window_days=14)

    # Per-user: for every event, is_relevant ∈ {0,1}, plus K/G/T scores.
    # Aggregate point-biserial correlation between each component and relevance.
    k_corr_list, g_corr_list, t_corr_list = [], [], []
    overlap_stats = {"k_nonzero": 0, "g_nonzero": 0, "t_nonzero": 0}

    for u in eligible:
        rel_set = truth[u]
        # Knowledge
        k_df = km.recommend(u, top_n=len(events))[["event_id", "KnowledgeScore"]]
        k_df["event_id"] = k_df["event_id"].astype(str)
        # Graph
        g_df = recommend_from_similar_users(
            attends=train, follows=follows, target_user=u, top_users=50, top_n=len(events)
        )
        if not g_df.empty:
            g_df["event_id"] = g_df["event_id"].astype(str)
        # Trend
        t_df = trend_df[["event_id", "TrendScore"]].copy()
        t_df["event_id"] = t_df["event_id"].astype(str)

        # Build a per-user table over the full event catalog
        merged = (
            pd.DataFrame({"event_id": events["event_id"].astype(str)})
            .merge(k_df, on="event_id", how="left")
            .merge(g_df, on="event_id", how="left")
            .merge(t_df, on="event_id", how="left")
        )
        merged[["KnowledgeScore", "GraphScore", "TrendScore"]] = merged[
            ["KnowledgeScore", "GraphScore", "TrendScore"]
        ].fillna(0.0)
        merged["relevant"] = merged["event_id"].isin(rel_set).astype(int)
        if merged["relevant"].sum() == 0:
            continue

        for col, lst, key in (
            ("KnowledgeScore", k_corr_list, "k_nonzero"),
            ("GraphScore", g_corr_list, "g_nonzero"),
            ("TrendScore", t_corr_list, "t_nonzero"),
        ):
            if merged[col].std() > 1e-9:
                corr = merged[[col, "relevant"]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    lst.append(corr)
                    overlap_stats[key] += int(merged[col].gt(0).sum())

    def stats(name: str, arr):
        if not arr:
            return {"name": name, "n": 0, "mean": None, "median": None, "p25": None, "p75": None}
        a = np.array(arr)
        return {
            "name": name,
            "n": int(len(a)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
        }

    out = {
        "users_with_correlation": {
            "knowledge": len(k_corr_list),
            "graph": len(g_corr_list),
            "trend": len(t_corr_list),
        },
        "correlation_with_relevance": {
            "knowledge": stats("knowledge", k_corr_list),
            "graph": stats("graph", g_corr_list),
            "trend": stats("trend", t_corr_list),
        },
        "interpretation": {
            "knowledge_mean": (
                "Near zero -> stated profile is decoupled from attendance (data issue). "
                "Positive -> Knowledge is informative; weighting issue or scoring still too uniform."
            ),
        },
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / "diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print()
    print("Per-user point-biserial correlation between component score and held-out relevance:")
    for k in ("knowledge", "graph", "trend"):
        s = out["correlation_with_relevance"][k]
        if s["n"] == 0:
            print(f"  {k:9s}: n=0 (no users had non-constant scores)")
        else:
            print(f"  {k:9s}: n={s['n']:3d}  mean={s['mean']:+.4f}  median={s['median']:+.4f}  IQR=[{s['p25']:+.4f}, {s['p75']:+.4f}]")

    print()
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    correlation_report()
