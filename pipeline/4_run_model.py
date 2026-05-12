"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Run the Actual Recommendation Model
=============================================================

Run:  python pipeline/4_run_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge_based import KnowledgeMatcher
from src.graph_based import recommend_from_similar_users, recommend_artists_from_similar_users
from src.trend_based import TrendWindowRecommender
from src.hybrid.hybrid_ranker import HybridRanker
from src.hybrid.explanations import attach_explanations
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, coverage
from src.data_io import load_users, load_events, load_attends, load_follows, load_artists

DATA_DIR = ROOT / "data"


def to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return {p.strip().strip("'\" ").lower() for p in s.split(",") if p.strip().strip("'\" ")}


def print_banner(text: str) -> None:
    w = 60
    print()
    print("=" * w)
    print(f"  {text}")
    print("=" * w)


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---\n")


def main() -> None:
    # ── Load data ───────────────────────────────────────────
    users = load_users(DATA_DIR)
    events = load_events(DATA_DIR)
    attends = load_attends(DATA_DIR)
    follows = load_follows(DATA_DIR)
    artists = load_artists(DATA_DIR)

    # ── Train / Test split (time-based) ─────────────────────
    attends["timestamp"] = pd.to_datetime(attends["timestamp"])
    attends = attends.sort_values("timestamp")
    test = attends.groupby("user_id").tail(1)
    train = attends.drop(test.index)

    # ── Select a user ────────────────────────────────────────
    user_counts = train.groupby("user_id").size().sort_values(ascending=False)
    print_banner("4: Run the Actual Recommendation Model")

    active = user_counts.head(20)
    print("    Active users (by training interactions):")
    for uid, cnt in active.items():
        uname = users.loc[users["user_id"] == uid, "name"].values
        uname = uname[0] if len(uname) else "?"
        print(f"      {uid}  ({cnt} interactions)  {uname}")
    print()
    chosen = input("    Enter a user_id from the list above: ").strip()
    if chosen not in set(users["user_id"]):
        print(f"    '{chosen}' not found — using {active.index[5]}")
        chosen = active.index[5]
    sample_user = chosen
    user_row = users.loc[users["user_id"] == sample_user].iloc[0]

    print_banner("4: Run the Actual Recommendation Model")

    print("    This section runs the FULL recommendation pipeline for a single user:")
    print("    1. Knowledge-Based  — matches user profile (interests, city) to event attributes")
    print("    2. Graph-Based      — finds similar users via shared attendance, recommends their events")
    print("    3. Trend-Based      — identifies events gaining popularity in a recent time window")
    print("    4. Hybrid Blend     — combines all three scores using dynamic weights")
    print("    5. Artist Recs      — profile matching + collaborative filtering for artist discovery")
    print("    6. Evaluation       — measures accuracy (Precision, Recall, NDCG, Coverage)")
    print()
    print("    PURPOSE: Show the complete end-to-end pipeline from raw data to ranked recommendations")
    print("    with human-readable explanations and offline accuracy metrics.")
    print()

    print_section(f"Selected User: {sample_user}")
    print(f"    Name             :  {user_row['name']}")
    print(f"    Art interests    :  {user_row.get('art_interests', 'N/A')}")
    print(f"    City             :  {user_row.get('city', 'N/A')}")
    print(f"    Train interactions:  {len(train.loc[train['user_id'] == sample_user])}")
    print(f"    Held-out event   :  {test.loc[test['user_id'] == sample_user, 'event_id'].iloc[0]}")

    TOP_N = 10

    # ── 1) Knowledge-based scores ───────────────────────────
    print_section("Model 1 — Knowledge-Based (Profile Matching)")    
    print("    HOW: Compares user's art_interests & city with each event's art_forms, genres & city")
    print("    WHY: Ensures recommendations match what the user is explicitly interested in")
    print()    
    km = KnowledgeMatcher(budget_col=None)
    km.fit(users, events)
    knowledge_df = km.recommend(sample_user, top_n=len(events))
    knowledge_scores = knowledge_df[["event_id", "KnowledgeScore"]]

    top_knowledge = knowledge_scores.sort_values("KnowledgeScore", ascending=False).head(TOP_N)
    for i, (_, r) in enumerate(top_knowledge.iterrows(), 1):
        ev_name = events.loc[events["event_id"] == r["event_id"], "name"].values
        name = ev_name[0] if len(ev_name) else "?"
        print(f"    {i:>2}. {r['event_id']}  {name:<40s}  K={r['KnowledgeScore']:.2f}")

    # ── 2) Graph-based scores ───────────────────────────────
    print_section("Model 2 — Graph-Based (Similar Users)")
    print("    HOW: Finds users who attended the same events (Jaccard similarity) and recommends THEIR events")
    print("    WHY: Leverages collective behaviour — 'users like you also enjoyed these events'")
    print()
    graph_df = recommend_from_similar_users(
        attends=train,
        follows=follows,
        target_user=sample_user,
        top_users=50,
        top_n=TOP_N * 3,
        alpha=0.5,
    )
    if graph_df.empty:
        graph_scores = pd.DataFrame(columns=["event_id", "GraphScore"])
        print("    (no graph recommendations — user may be cold-start)")
    else:
        graph_scores = graph_df[["event_id", "GraphScore"]]
        top_graph = graph_scores.head(TOP_N)
        for i, (_, r) in enumerate(top_graph.iterrows(), 1):
            ev_name = events.loc[events["event_id"] == r["event_id"], "name"].values
            name = ev_name[0] if len(ev_name) else "?"
            print(f"    {i:>2}. {r['event_id']}  {name:<40s}  G={r['GraphScore']:.3f}")

    # ── 3) Trend-based scores ───────────────────────────────
    print_section("Model 3 — Trend-Based (Recent Popularity)")
    print("    HOW: Counts attendance in a 14-day window and measures growth rate")
    print("    WHY: Captures momentum — events gaining popularity may be relevant right now")
    print()
    trend_model = TrendWindowRecommender().fit(train)
    trend_df = trend_model.recommend(top_n=TOP_N * 3, window_days=14)
    trend_scores = trend_df[["event_id", "TrendScore"]]

    top_trend = trend_scores.head(TOP_N)
    for i, (_, r) in enumerate(top_trend.iterrows(), 1):
        ev_name = events.loc[events["event_id"] == r["event_id"], "name"].values
        name = ev_name[0] if len(ev_name) else "?"
        print(f"    {i:>2}. {r['event_id']}  {name:<40s}  T={r['TrendScore']:.3f}")

    # ── 4) Hybrid blend ─────────────────────────────────────
    print_section("Hybrid Model — Weighted Combination")
    print("    HOW: FinalScore = α·Knowledge + β·Graph + γ·Trend (weights depend on user activity)")
    print("    WHY: No single model is best for all users — cold-start users rely more on knowledge,")
    print("         active users benefit from collaborative filtering, trending boosts timely events")
    print()
    candidates = pd.DataFrame({"event_id": pd.unique(
        pd.concat([
            knowledge_scores["event_id"],
            graph_scores.get("event_id", pd.Series(dtype=str)),
            trend_scores.get("event_id", pd.Series(dtype=str)),
        ], ignore_index=True)
    )})
    merged = candidates.merge(knowledge_scores, on="event_id", how="left")
    merged = merged.merge(graph_scores, on="event_id", how="left")
    merged = merged.merge(trend_scores, on="event_id", how="left")
    merged[["KnowledgeScore", "GraphScore", "TrendScore"]] = merged[
        ["KnowledgeScore", "GraphScore", "TrendScore"]
    ].fillna(0.0)

    user_interactions = len(train.loc[train["user_id"] == sample_user])
    ranker = HybridRanker()
    ranked = ranker.rank(merged, user_interactions=user_interactions, top_n=TOP_N)

    strategy = "cold_start" if user_interactions < ranker.interaction_threshold else "active"
    scheme = ranker.weights[strategy]
    print(f"    Strategy selected :  {strategy}")
    print(f"    Weights           :  α={scheme.alpha}  β={scheme.beta}  γ={scheme.gamma}")
    print()

    # Attach explanations
    interests = to_tokens(user_row.get("art_interests"))
    city = user_row.get("city") if pd.notna(user_row.get("city")) else None
    ranked = attach_explanations(ranked, events=events, user_interests=interests, user_city=city)

    print(f"    {'Rank':<5} {'Event':<8} {'Name':<38} {'K':>5} {'G':>7} {'T':>7} {'Final':>7}  Explanation")
    print("    " + "-" * 110)
    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        ev_name = events.loc[events["event_id"] == r["event_id"], "name"].values
        name = (ev_name[0] if len(ev_name) else "?")[:37]
        expl = "; ".join(r["Explanations"])
        print(
            f"    {i:<5} {r['event_id']:<8} {name:<38} "
            f"{r['KnowledgeScore']:>5.2f} {r['GraphScore']:>7.3f} {r['TrendScore']:>7.3f} "
            f"{r['FinalScore']:>7.3f}  {expl}"
        )

    # ── 5) Artist recommendations ───────────────────────────
    print_section("Artist Recommendations")
    print("    HOW: Profile matching (user interests vs artist art_forms/genres)")
    print("         + Collaborative (artists followed by similar users)")
    print("    WHY: Artists are a natural discovery axis — finding new artists leads to new events")
    print()
    print("    Combining profile matching + collaborative filtering for artists")
    print()

    followed_ids = set(follows.loc[follows["user_id"] == sample_user, "artist_id"])
    user_interests = to_tokens(user_row.get("art_interests", ""))

    # A) Profile-matched artists
    artist_profile: dict[str, tuple[float, list[str]]] = {}
    for _, art in artists.iterrows():
        aid = art["artist_id"]
        if aid in followed_ids:
            continue
        art_cats: set[str] = set()
        for col in ("art_forms", "genres"):
            art_cats.update(to_tokens(art.get(col)))
        overlap = user_interests & art_cats
        if overlap and user_interests:
            artist_profile[aid] = (len(overlap) / len(user_interests), sorted(overlap))

    profile_recs = sorted(artist_profile.items(), key=lambda kv: kv[1][0], reverse=True)[:TOP_N]

    # B) Collaborative artist recs
    collab_artist_df = recommend_artists_from_similar_users(
        attends=train, follows=follows,
        target_user=sample_user, top_users=50, top_n=TOP_N, alpha=0.5,
    )

    # Merge scores
    artist_final: dict[str, dict] = {}
    for aid, (score, cats) in profile_recs:
        artist_final[aid] = {"artist_id": aid, "ProfileScore": score, "CollabScore": 0.0, "shared": cats}
    for _, row in collab_artist_df.iterrows():
        aid = row["artist_id"]
        if aid in artist_final:
            artist_final[aid]["CollabScore"] = row["ArtistGraphScore"]
        else:
            artist_final[aid] = {"artist_id": aid, "ProfileScore": 0.0, "CollabScore": row["ArtistGraphScore"], "shared": []}

    for d in artist_final.values():
        d["FinalArtistScore"] = 0.5 * d["ProfileScore"] + 0.5 * d["CollabScore"]

    artist_ranked = sorted(artist_final.values(), key=lambda d: d["FinalArtistScore"], reverse=True)[:TOP_N]

    print(f"    {'#':<3} {'Artist':<8} {'Name':<38s} {'Prof':>5} {'Collab':>7} {'Final':>7}  Shared Categories")
    print("    " + "-" * 100)
    for i, rec in enumerate(artist_ranked, 1):
        aid = rec["artist_id"]
        art_name = artists.loc[artists["artist_id"] == aid, "name"]
        name = (art_name.iloc[0] if not art_name.empty else "?")[:37]
        shared = ", ".join(rec["shared"][:3]) if rec["shared"] else "-"
        print(
            f"    {i:<3} {aid:<8} {name:<38s} "
            f"{rec['ProfileScore']:>5.2f} {rec['CollabScore']:>7.3f} {rec['FinalArtistScore']:>7.3f}  {shared}"
        )
    print()

    # ── 6) Accuracy metrics ─────────────────────────────────
    print_section("Accuracy Metrics (Offline Evaluation)")
    print("    HOW: Hold out each user's last interaction, run the model on the rest,")
    print("         check if the held-out event appears in the top-N recommendations")
    print("    METRICS:")
    print("      • Precision@K  = fraction of top-K recs that are actually relevant")
    print("      • Recall@K     = fraction of relevant items found in top-K")
    print("      • NDCG@K       = rewards placing relevant items higher in the list")
    print("      • Coverage     = fraction of catalog recommended to at least one user")
    print()

    # Evaluate across a sample of users with holdout
    rec_map: dict[str, list[str]] = {}
    rel_map: dict[str, set[str]] = {}

    test_users = test["user_id"].unique()
    sample_eval_users = test_users[:30]  # small sample for speed

    print(f"    Evaluating {len(sample_eval_users)} users ...", end="", flush=True)

    for idx, uid in enumerate(sample_eval_users):
        rel_map[uid] = set(test.loc[test["user_id"] == uid, "event_id"].astype(str))

        try:
            km_df = km.recommend(uid, top_n=TOP_N * 3)
            k_scores = km_df[["event_id", "KnowledgeScore"]]
        except Exception:
            k_scores = pd.DataFrame(columns=["event_id", "KnowledgeScore"])

        try:
            g_df = recommend_from_similar_users(train, follows, uid, top_users=20, top_n=TOP_N * 2, alpha=0.5)
            g_scores = g_df[["event_id", "GraphScore"]] if not g_df.empty else pd.DataFrame(columns=["event_id", "GraphScore"])
        except Exception:
            g_scores = pd.DataFrame(columns=["event_id", "GraphScore"])

        cand = pd.DataFrame({"event_id": pd.unique(
            pd.concat([k_scores["event_id"], g_scores.get("event_id", pd.Series(dtype=str)), trend_scores.get("event_id", pd.Series(dtype=str))], ignore_index=True)
        )})
        m = cand.merge(k_scores, on="event_id", how="left")
        m = m.merge(g_scores, on="event_id", how="left")
        m = m.merge(trend_scores, on="event_id", how="left")
        m[["KnowledgeScore", "GraphScore", "TrendScore"]] = m[["KnowledgeScore", "GraphScore", "TrendScore"]].fillna(0.0)

        ui = len(train.loc[train["user_id"] == uid])
        r = ranker.rank(m, user_interactions=ui, top_n=TOP_N)
        rec_map[uid] = r["event_id"].astype(str).tolist()

        if (idx + 1) % 10 == 0:
            print(f" {idx + 1}", end="", flush=True)

    print(" done.")
    catalog = set(events["event_id"].astype(str))

    avg_p = sum(precision_at_k(rec_map[u], rel_map[u], TOP_N) for u in sample_eval_users) / len(sample_eval_users)
    avg_r = sum(recall_at_k(rec_map[u], rel_map[u], TOP_N) for u in sample_eval_users) / len(sample_eval_users)
    avg_n = sum(ndcg_at_k(rec_map[u], rel_map[u], TOP_N) for u in sample_eval_users) / len(sample_eval_users)
    cov = coverage(rec_map, catalog)

    print(f"    Evaluated users     :  {len(sample_eval_users)}")
    print(f"    Precision@{TOP_N:<3}      :  {avg_p:.4f}")
    print(f"    Recall@{TOP_N:<3}         :  {avg_r:.4f}")
    print(f"    NDCG@{TOP_N:<3}           :  {avg_n:.4f}")
    print(f"    Catalog coverage    :  {cov:.4f}  ({int(cov * len(catalog))}/{len(catalog)} events)")

    print_banner("End of Section 4")
    print()


if __name__ == "__main__":
    main()
