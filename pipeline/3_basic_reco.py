"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Basic Recommendation Analysis
=============================================================

Run:  python pipeline/3_basic_reco.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

DATA_DIR = Path("data")

USER_COLOR = "#4A90D9"
EVENT_COLOR = "#E8A838"
CATEGORY_COLOR = "#50C878"
RECO_COLOR = "#E05555"
ARTIST_COLOR = "#FF8C42"


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
    # ── Load ────────────────────────────────────────────────
    users = pd.read_csv(DATA_DIR / "users.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    artists = pd.read_csv(DATA_DIR / "artists.csv")
    follows = pd.read_csv(DATA_DIR / "follows.csv")

    # ── Select a user ────────────────────────────────────────
    user_counts = attends.groupby("user_id").size().sort_values(ascending=False)
    print_banner("3: Basic Recommendation Analysis")

    active = user_counts.head(20)
    print("    Active users (by attendance count):")
    for uid, cnt in active.items():
        uname = users.loc[users["user_id"] == uid, "name"].values
        uname = uname[0] if len(uname) else "?"
        print(f"      {uid}  ({cnt} events)  {uname}")
    print()
    chosen = input("    Enter a user_id from the list above: ").strip()
    if chosen not in set(users["user_id"]):
        print(f"    '{chosen}' not found — using {active.index[5]}")
        chosen = active.index[5]
    sample_user = chosen
    user_row = users.loc[users["user_id"] == sample_user].iloc[0]

    print_section(f"Selected User: {sample_user}")
    print(f"    Name             :  {user_row['name']}")
    print(f"    Art interests    :  {user_row.get('art_interests', 'N/A')}")
    print(f"    City             :  {user_row.get('city', 'N/A')}")
    print(f"    Culture prefs    :  {user_row.get('culture_preferences', 'N/A')}")

    # ── Items already interacted with ───────────────────────
    attended_ids = attends.loc[attends["user_id"] == sample_user, "event_id"].tolist()

    print_section("Events Already Attended")
    attended_events = events.loc[events["event_id"].isin(attended_ids)].head(6)
    for _, row in attended_events.iterrows():
        print(f"    {row['event_id']}  {row['name']:<40s}  {row.get('art_forms', '')}")
    print(f"    ... total attended: {len(attended_ids)}")

    # ── Related categories ──────────────────────────────────
    user_cats: set[str] = set()
    for _, row in attended_events.iterrows():
        for col in ("art_forms", "genres"):
            user_cats.update(to_tokens(row.get(col)))

    print_section("Categories Linked to Attended Events")
    print(f"    {', '.join(sorted(user_cats))}")

    # ── Method 1: Category-Based Recommendation ─────────────
    # Find events the user has NOT attended that share categories
    print_section("Method 1 — Category Path Recommendations")
    print("    Logic: User → Attended Event → Category → New Event")
    print()

    candidate_scores: dict[str, tuple[float, list[str]]] = {}
    attended_set = set(attended_ids)

    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid in attended_set:
            continue
        ev_cats: set[str] = set()
        for col in ("art_forms", "genres"):
            ev_cats.update(to_tokens(ev.get(col)))
        overlap = user_cats & ev_cats
        if overlap:
            score = len(overlap) / len(user_cats) if user_cats else 0
            candidate_scores[eid] = (score, sorted(overlap))

    cat_recs = sorted(candidate_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:5]

    for rank, (eid, (score, cats)) in enumerate(cat_recs, 1):
        ev_name = events.loc[events["event_id"] == eid, "name"].iloc[0]
        shared = ", ".join(cats)
        print(f"    {rank}. {eid}  {ev_name:<40s}  score={score:.2f}")
        print(f"       Path: {sample_user} → attended events → [{shared}] → {eid}")
    print()

    # ── Method 2: Similar-User Recommendation ────────────────
    print_section("Method 2 — Similar User Recommendations")
    print("    Logic: User A → shared events → User B → User B's other events")
    print()

    # Find users who attended the same events
    co_users = (
        attends.loc[attends["event_id"].isin(attended_set) & (attends["user_id"] != sample_user)]
        .groupby("user_id")["event_id"]
        .apply(set)
    )
    sim_scores: dict[str, float] = {}
    for uid, their_events in co_users.items():
        overlap_count = len(their_events & attended_set)
        union_count = len(their_events | attended_set)
        if union_count:
            sim_scores[uid] = overlap_count / union_count

    top_sim_users = sorted(sim_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]
    sim_user_ids = [u for u, _ in top_sim_users]

    # Events attended by similar users but not by our user
    sim_event_scores: dict[str, float] = {}
    for uid, sim in top_sim_users:
        their_events = set(attends.loc[attends["user_id"] == uid, "event_id"])
        new_events = their_events - attended_set
        for eid in new_events:
            sim_event_scores[eid] = sim_event_scores.get(eid, 0.0) + sim

    sim_recs = sorted(sim_event_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

    for rank, (eid, score) in enumerate(sim_recs, 1):
        ev_name = events.loc[events["event_id"] == eid, "name"].iloc[0]
        print(f"    {rank}. {eid}  {ev_name:<40s}  score={score:.3f}")
        print(f"       Path: {sample_user} → shared events → similar users → {eid}")
    print()

    # ── Method 3: Artist Recommendations ─────────────────────
    print_section("Method 3 — Artist Recommendations")
    print("    Logic A: Profile matching — user art_interests vs artist art_forms/genres")
    print("    Logic B: Collaborative — artists followed by similar users")
    print()

    followed_ids = set(follows.loc[follows["user_id"] == sample_user, "artist_id"])
    user_interests = to_tokens(user_row.get("art_interests", ""))

    # A) Profile-matched artists
    artist_profile_scores: dict[str, tuple[float, list[str]]] = {}
    for _, art in artists.iterrows():
        aid = art["artist_id"]
        if aid in followed_ids:
            continue
        art_cats = set()
        for col in ("art_forms", "genres"):
            art_cats.update(to_tokens(art.get(col)))
        overlap = user_interests & art_cats
        if overlap and user_interests:
            artist_profile_scores[aid] = (len(overlap) / len(user_interests), sorted(overlap))

    profile_artist_recs = sorted(artist_profile_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:5]

    print("    A) Profile-Matched Artists:")
    for rank, (aid, (score, cats)) in enumerate(profile_artist_recs, 1):
        art_name = artists.loc[artists["artist_id"] == aid, "name"].iloc[0]
        shared = ", ".join(cats)
        print(f"    {rank}. {aid}  {art_name:<40s}  score={score:.2f}  [{shared}]")
    print()

    # B) Artists followed by similar users (but not by this user)
    sim_artist_scores: dict[str, float] = {}
    for uid, sim in top_sim_users[:10]:
        their_artists = set(follows.loc[follows["user_id"] == uid, "artist_id"])
        new_artists = their_artists - followed_ids
        for aid in new_artists:
            sim_artist_scores[aid] = sim_artist_scores.get(aid, 0.0) + sim

    collab_artist_recs = sorted(sim_artist_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

    print("    B) Artists Followed by Similar Users:")
    for rank, (aid, score) in enumerate(collab_artist_recs, 1):
        art_name = artists.loc[artists["artist_id"] == aid, "name"]
        name = art_name.iloc[0] if not art_name.empty else "?"
        print(f"    {rank}. {aid}  {name:<40s}  score={score:.3f}")
        print(f"       Path: {sample_user} → shared events → similar users → follows → {aid}")
    print()

    # ── Visualise recommendation paths ───────────────────────
    print_section("Graph Visualization")
    print("    This 3-panel graph shows HOW recommendations are made:")
    print("")
    print("    LEFT   — Category matching: finds events sharing categories with what you attended")
    print("    MIDDLE — Collaborative filtering: finds events that similar users enjoyed")
    print("    RIGHT  — Artist discovery: finds artists matching your interests + similar users' follows")
    print("")
    print("    Each colored path traces the logic from YOU → to a specific recommendation.")
    print("    Dashed edges show the final recommendation step.")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        f"Three Recommendation Strategies for {sample_user}\n" +
        "Visualizing the path from user profile to final recommendations",
        fontsize=13,
        fontweight="bold",
    )

    # --- Left panel: category path ---
    ax = axes[0]
    ax.set_title(
        "Method 1: Category-Path Logic\n" +
        "YOU → attended events → shared categories → NEW events",
        fontsize=10,
        fontweight="bold",
    )

    G1 = nx.DiGraph()
    G1.add_node(sample_user, kind="user")
    # Add a few attended events
    shown_attended = list(attended_set)[:3]
    for eid in shown_attended:
        G1.add_node(eid, kind="event")
        G1.add_edge(sample_user, eid, relation="attended")

    # Gather categories from shown attended
    shown_cats: set[str] = set()
    for eid in shown_attended:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        row = row.iloc[0]
        for col in ("art_forms", "genres"):
            shown_cats.update(to_tokens(row.get(col)))

    for cat in list(shown_cats)[:4]:
        cn = f"cat:{cat}"
        G1.add_node(cn, kind="category")
        for eid in shown_attended:
            r = events.loc[events["event_id"] == eid]
            if r.empty:
                continue
            ev_cats = set()
            for col in ("art_forms", "genres"):
                ev_cats.update(to_tokens(r.iloc[0].get(col)))
            if cat in ev_cats:
                G1.add_edge(eid, cn, relation="belongs_to")

    # Add recommended events
    for eid, (score, cats) in cat_recs[:3]:
        G1.add_node(eid, kind="recommended")
        for cat in cats[:2]:
            cn = f"cat:{cat}"
            if cn in G1:
                G1.add_edge(cn, eid, relation="recommends")

    pos1 = nx.spring_layout(G1, seed=42, k=2.0)
    for kind, color, size in [
        ("user", USER_COLOR, 800),
        ("event", EVENT_COLOR, 500),
        ("category", CATEGORY_COLOR, 400),
        ("recommended", RECO_COLOR, 600),
    ]:
        nodes = [n for n, d in G1.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(G1, pos1, nodelist=nodes, node_color=color, node_size=size, edgecolors="white", linewidths=1.2, ax=ax)

    normal_edges = [(u, v) for u, v, d in G1.edges(data=True) if d.get("relation") != "recommends"]
    reco_edges = [(u, v) for u, v, d in G1.edges(data=True) if d.get("relation") == "recommends"]
    nx.draw_networkx_edges(G1, pos1, edgelist=normal_edges, edge_color="#999", width=1.2, alpha=0.5, arrows=True, ax=ax)
    nx.draw_networkx_edges(G1, pos1, edgelist=reco_edges, edge_color=RECO_COLOR, width=2.0, alpha=0.8, style="dashed", arrows=True, ax=ax)

    labels1 = {n: n.replace("cat:", "") for n in G1.nodes()}
    nx.draw_networkx_labels(G1, pos1, labels1, font_size=7, font_weight="bold", ax=ax)
    
    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Strategy: If you attended music events, recommend other music events\n" +
        "you haven't seen yet (content-based filtering)",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
    )
    ax.axis("off")

    # --- Middle panel: similar-user path ---
    ax = axes[1]
    ax.set_title(
        "Method 2: Collaborative Filtering\n" +
        "YOU → shared events ← similar users → THEIR other events",
        fontsize=10,
        fontweight="bold",
    )

    G2 = nx.DiGraph()
    G2.add_node(sample_user, kind="user")

    shared_events_shown = list(attended_set)[:2]
    for eid in shared_events_shown:
        G2.add_node(eid, kind="event")
        G2.add_edge(sample_user, eid, relation="attended")

    for uid, sim in top_sim_users[:2]:
        G2.add_node(uid, kind="similar_user")
        for eid in shared_events_shown:
            their = set(attends.loc[attends["user_id"] == uid, "event_id"])
            if eid in their:
                G2.add_edge(uid, eid, relation="attended")

    for eid, score in sim_recs[:3]:
        G2.add_node(eid, kind="recommended")
        for uid, _ in top_sim_users[:2]:
            their = set(attends.loc[attends["user_id"] == uid, "event_id"])
            if eid in their:
                G2.add_edge(uid, eid, relation="recommends")

    pos2 = nx.spring_layout(G2, seed=99, k=2.0)
    for kind, color, size in [
        ("user", USER_COLOR, 800),
        ("event", EVENT_COLOR, 500),
        ("similar_user", "#9B59B6", 600),
        ("recommended", RECO_COLOR, 600),
    ]:
        nodes = [n for n, d in G2.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(G2, pos2, nodelist=nodes, node_color=color, node_size=size, edgecolors="white", linewidths=1.2, ax=ax)

    normal_edges2 = [(u, v) for u, v, d in G2.edges(data=True) if d.get("relation") != "recommends"]
    reco_edges2 = [(u, v) for u, v, d in G2.edges(data=True) if d.get("relation") == "recommends"]
    nx.draw_networkx_edges(G2, pos2, edgelist=normal_edges2, edge_color="#999", width=1.2, alpha=0.5, arrows=True, ax=ax)
    nx.draw_networkx_edges(G2, pos2, edgelist=reco_edges2, edge_color=RECO_COLOR, width=2.0, alpha=0.8, style="dashed", arrows=True, ax=ax)

    nx.draw_networkx_labels(G2, pos2, font_size=7, font_weight="bold", ax=ax)
    
    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Strategy: Find users with similar taste (shared event attendance),\n" +
        "then recommend events THEY enjoyed (social proof)",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )
    ax.axis("off")

    # --- Right panel: artist recommendations ---
    ax = axes[2]
    ax.set_title(
        "Method 3: Artist Discovery\n" +
        "YOU → interests/categories → matching artists + similar users' follows",
        fontsize=10,
        fontweight="bold",
    )

    G3 = nx.DiGraph()
    G3.add_node(sample_user, kind="user")

    # Add user interest categories as nodes
    for cat in list(user_interests)[:4]:
        cn = f"cat:{cat}"
        G3.add_node(cn, kind="category")
        G3.add_edge(sample_user, cn, relation="interested_in")

    # Profile-matched artist nodes
    for aid, (score, cats) in profile_artist_recs[:3]:
        G3.add_node(aid, kind="artist")
        for cat in cats[:2]:
            cn = f"cat:{cat}"
            if cn in G3:
                G3.add_edge(cn, aid, relation="recommends")

    # Collaborative artist nodes
    for aid, score in collab_artist_recs[:3]:
        if aid not in G3:
            G3.add_node(aid, kind="artist")
        # Link via similar users
        for uid, _ in top_sim_users[:2]:
            their_artists = set(follows.loc[follows["user_id"] == uid, "artist_id"])
            if aid in their_artists:
                if uid not in G3:
                    G3.add_node(uid, kind="similar_user")
                    G3.add_edge(sample_user, uid, relation="similar_to")
                G3.add_edge(uid, aid, relation="recommends")
                break

    pos3 = nx.spring_layout(G3, seed=77, k=2.0)
    for kind, color, size in [
        ("user", USER_COLOR, 800),
        ("category", CATEGORY_COLOR, 400),
        ("similar_user", "#9B59B6", 600),
        ("artist", ARTIST_COLOR, 600),
    ]:
        nodes = [n for n, d in G3.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(G3, pos3, nodelist=nodes, node_color=color, node_size=size, edgecolors="white", linewidths=1.2, ax=ax)

    normal_edges3 = [(u, v) for u, v, d in G3.edges(data=True) if d.get("relation") != "recommends"]
    reco_edges3 = [(u, v) for u, v, d in G3.edges(data=True) if d.get("relation") == "recommends"]
    nx.draw_networkx_edges(G3, pos3, edgelist=normal_edges3, edge_color="#999", width=1.2, alpha=0.5, arrows=True, ax=ax)
    nx.draw_networkx_edges(G3, pos3, edgelist=reco_edges3, edge_color=ARTIST_COLOR, width=2.0, alpha=0.8, style="dashed", arrows=True, ax=ax)

    labels3 = {n: n.replace("cat:", "") for n in G3.nodes()}
    nx.draw_networkx_labels(G3, pos3, labels3, font_size=7, font_weight="bold", ax=ax)
    
    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Strategy: Match your stated interests to artist profiles (profile matching)\n" +
        "+ discover artists that similar users follow (collaborative)",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="peachpuff", alpha=0.7),
    )
    ax.axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=USER_COLOR, label="Target user"),
        mpatches.Patch(color=EVENT_COLOR, label="Attended event"),
        mpatches.Patch(color=CATEGORY_COLOR, label="Category"),
        mpatches.Patch(color="#9B59B6", label="Similar user"),
        mpatches.Patch(color=RECO_COLOR, label="Recommended event"),
        mpatches.Patch(color=ARTIST_COLOR, label="Recommended artist"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_path = Path("pipeline") / "3_basic_reco.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    Graph saved to:  {out_path}")

    print_banner("End of Section 3")
    print()
    plt.show()


if __name__ == "__main__":
    main()
