"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Highlight Recommendation Paths
=============================================================

For each recommended event we trace the path through the graph
that leads from the target user to the recommendation.

Two path families are shown:
  A)  User ─attended─➤ Event ─belongs_to─➤ Category ─belongs_to─➤ Recommended Event
  B)  User ─attended─➤ Event ←─attended─ Similar User ─attended─➤ Recommended Event

Run:  python pipeline/5_highlight_paths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_io import load_users, load_events, load_attends, load_follows, load_artists  # noqa: E402

DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────
USER_COLOR = "#4A90D9"
EVENT_COLOR = "#E8A838"
CATEGORY_COLOR = "#50C878"
RECO_COLOR = "#E05555"
SIM_USER_COLOR = "#9B59B6"
ARTIST_COLOR = "#FF8C42"
PATH_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
FADED = "#D0D0D0"


def to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return {
        p.strip().strip("'\" ").lower()
        for p in s.split(",")
        if p.strip().strip("'\" ")
    }


def print_banner(text: str) -> None:
    w = 60
    print()
    print("=" * w)
    print(f"  {text}")
    print("=" * w)


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---\n")


# ─────────────────────────────────────────────────────────────
def main() -> None:
    users = load_users(DATA_DIR)
    events = load_events(DATA_DIR)
    attends = load_attends(DATA_DIR)
    artists = load_artists(DATA_DIR)
    follows = load_follows(DATA_DIR)

    # ── Select a user ────────────────────────────────────────
    user_counts = attends.groupby("user_id").size().sort_values(ascending=False)
    print_banner("5: Highlight Recommendation Paths")

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

    print_section(f"Target User: {sample_user}")
    print(f"    Name           :  {user_row['name']}")
    print(f"    Art interests  :  {user_row.get('art_interests', 'N/A')}")

    attended_ids = set(
        attends.loc[attends["user_id"] == sample_user, "event_id"]
    )

    # ── Build lightweight heterogeneous graph ────────────────
    G = nx.Graph()
    G.add_node(sample_user, kind="user")
    for eid in attended_ids:
        G.add_node(eid, kind="event")
        G.add_edge(sample_user, eid, relation="attended")

    # Categories for attended events
    for eid in attended_ids:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        for col in ("art_forms", "genres"):
            for cat in to_tokens(row.iloc[0].get(col)):
                cn = f"cat:{cat}"
                if cn not in G:
                    G.add_node(cn, kind="category")
                G.add_edge(eid, cn, relation="belongs_to")

    # Add candidate (non-attended) events linked to same categories
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid in attended_ids:
            continue
        for col in ("art_forms", "genres"):
            for cat in to_tokens(ev.get(col)):
                cn = f"cat:{cat}"
                if cn in G:
                    if eid not in G:
                        G.add_node(eid, kind="candidate")
                    G.add_edge(eid, cn, relation="belongs_to")

    # Similar users (co-attendance)
    co_users = (
        attends.loc[
            attends["event_id"].isin(attended_ids)
            & (attends["user_id"] != sample_user)
        ]
        .groupby("user_id")["event_id"]
        .apply(set)
    )
    sim_scores: dict[str, float] = {}
    for uid, their in co_users.items():
        j = len(their & attended_ids) / len(their | attended_ids)
        if j > 0:
            sim_scores[uid] = j
    top_sim = sorted(sim_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

    for uid, sim in top_sim:
        G.add_node(uid, kind="similar_user")
        their_events = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for eid in their_events:
            if eid in attended_ids:
                G.add_edge(uid, eid, relation="attended")
            elif eid in G:
                G.add_edge(uid, eid, relation="attended")
            else:
                G.add_node(eid, kind="candidate")
                G.add_edge(uid, eid, relation="attended")

    # ── Category-path recommendations ────────────────────────
    cat_user = set()
    for eid in attended_ids:
        r = events.loc[events["event_id"] == eid]
        if r.empty:
            continue
        for col in ("art_forms", "genres"):
            cat_user.update(to_tokens(r.iloc[0].get(col)))

    cat_scores: dict[str, tuple[float, list[str]]] = {}
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid in attended_ids:
            continue
        ev_cats = set()
        for col in ("art_forms", "genres"):
            ev_cats.update(to_tokens(ev.get(col)))
        overlap = cat_user & ev_cats
        if overlap:
            cat_scores[eid] = (len(overlap) / len(cat_user), sorted(overlap))

    cat_recs = sorted(cat_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:3]

    # ── Similar-user recommendations ─────────────────────────
    sim_event_scores: dict[str, tuple[float, str]] = {}
    for uid, sim in top_sim:
        their = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for eid in their - attended_ids:
            if eid not in sim_event_scores or sim > sim_event_scores[eid][0]:
                sim_event_scores[eid] = (sim, uid)
    sim_recs = sorted(sim_event_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:3]

    # ── Print traced paths ───────────────────────────────────
    print_section("Category-Path Traces (User → Event → Category → Rec)")
    for rank, (eid, (score, cats)) in enumerate(cat_recs, 1):
        ev_name = events.loc[events["event_id"] == eid, "name"].iloc[0]
        bridge_event = None
        for ae in attended_ids:
            r = events.loc[events["event_id"] == ae]
            if r.empty:
                continue
            ae_cats = set()
            for col in ("art_forms", "genres"):
                ae_cats.update(to_tokens(r.iloc[0].get(col)))
            if ae_cats & set(cats):
                bridge_event = ae
                break
        shared = cats[0] if cats else "?"
        print(f"    {rank}. {ev_name}  (score={score:.2f})")
        print(f"       {sample_user} ──attended──-> {bridge_event}")
        print(f"       {bridge_event} ──belongs_to──-> cat:{shared}")
        print(f"       cat:{shared} ──belongs_to──-> {eid}")
        print()

    print_section("Similar-User Path Traces (User → Event ← SimUser → Rec)")
    for rank, (eid, (sim, via_user)) in enumerate(sim_recs, 1):
        ev_name = events.loc[events["event_id"] == eid, "name"].iloc[0]
        shared_event = None
        via_attended = set(attends.loc[attends["user_id"] == via_user, "event_id"])
        for ae in attended_ids & via_attended:
            shared_event = ae
            break
        print(f"    {rank}. {ev_name}  (similarity={sim:.3f})")
        print(f"       {sample_user} ──attended──-> {shared_event}")
        print(f"       {via_user} ──attended──-> {shared_event}  (shared)")
        print(f"       {via_user} ──attended──-> {eid}  (recommendation)")
        print()
    # ── Artist path traces ─────────────────────────────────
    followed_ids = set(follows.loc[follows["user_id"] == sample_user, "artist_id"])
    user_interests = to_tokens(user_row.get("art_interests", ""))

    # Profile-matched artist recs
    artist_profile_scores: dict[str, tuple[float, list[str]]] = {}
    for _, art in artists.iterrows():
        aid = art["artist_id"]
        if aid in followed_ids:
            continue
        art_cats: set[str] = set()
        for col in ("art_forms", "genres"):
            art_cats.update(to_tokens(art.get(col)))
        overlap = user_interests & art_cats
        if overlap and user_interests:
            artist_profile_scores[aid] = (len(overlap) / len(user_interests), sorted(overlap))

    profile_artist_recs = sorted(artist_profile_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:3]

    # Collaborative artist recs
    sim_artist_scores: dict[str, tuple[float, str]] = {}
    for uid, sim in top_sim:
        their_artists = set(follows.loc[follows["user_id"] == uid, "artist_id"])
        for aid in their_artists - followed_ids:
            if aid not in sim_artist_scores or sim > sim_artist_scores[aid][0]:
                sim_artist_scores[aid] = (sim, uid)
    collab_artist_recs = sorted(sim_artist_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:3]

    print_section("Artist Path Traces (Profile → Category → Artist)")
    for rank, (aid, (score, cats)) in enumerate(profile_artist_recs, 1):
        art_name = artists.loc[artists["artist_id"] == aid, "name"]
        name = art_name.iloc[0] if not art_name.empty else "?"
        shared = cats[0] if cats else "?"
        print(f"    {rank}. {name}  (score={score:.2f})")
        print(f"       {sample_user} ──interests──➤ [{shared}]")
        print(f"       [{shared}] ──matches──➤ {aid}")
        print()

    print_section("Artist Path Traces (User → SimUser → follows → Artist)")
    for rank, (aid, (sim, via_user)) in enumerate(collab_artist_recs, 1):
        art_name = artists.loc[artists["artist_id"] == aid, "name"]
        name = art_name.iloc[0] if not art_name.empty else "?"
        shared_event = None
        via_attended = set(attends.loc[attends["user_id"] == via_user, "event_id"])
        for ae in attended_ids & via_attended:
            shared_event = ae
            break
        print(f"    {rank}. {name}  (similarity={sim:.3f})")
        print(f"       {sample_user} ──attended──➤ {shared_event}  (shared)")
        print(f"       {via_user} ──attended──➤ {shared_event}  (shared)")
        print(f"       {via_user} ──follows──➤ {aid}")
        print()
    # ── Visualise highlighted paths ──────────────────────────
    print_section("Graph Visualization — Path Highlighting")
    print("    This 3-panel graph traces the EXACT PATH from YOU to each recommendation:")
    print()
    print("    LEFT   — Content-based paths: traces YOUR events → shared categories → NEW events")
    print("             Each colored line = one recommendation's reasoning chain")
    print("    MIDDLE — Collaborative paths: traces YOU ← shared events → similar users → THEIR events")
    print("             Shows which similar user 'bridged' you to each recommendation")
    print("    RIGHT  — Artist discovery paths: traces YOUR interests → categories → artists")
    print("             + similar users → artists THEY follow (collaborative)")
    print()
    print("    WHY THIS MATTERS: Unlike Section 3 which shows the logic,")
    print("    this graph highlights the specific evidence trail for each recommendation.")
    print("    You can point to any colored path and explain exactly WHY that item was recommended.")
    print()

    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    fig.suptitle(
        f"Recommendation Path Traces for {sample_user}\n"
        + "Each colored line traces the exact evidence chain from YOU to a recommendation",
        fontsize=14,
        fontweight="bold",
    )

    # ---- LEFT: Category paths ----
    ax = axes[0]
    ax.set_title(
        "Path Type A: Content-Based Filtering\n"
        + "YOU → attended event → shared category → NEW event",
        fontsize=10, fontweight="bold",
    )

    Gc = nx.DiGraph()
    Gc.add_node(sample_user, kind="user")

    path_node_sets: list[set] = []
    path_edge_sets: list[list[tuple]] = []

    for idx, (eid, (score, cats)) in enumerate(cat_recs):
        ev_name_short = str(eid)
        Gc.add_node(eid, kind="recommended")

        bridge_event = None
        for ae in attended_ids:
            r = events.loc[events["event_id"] == ae]
            if r.empty:
                continue
            ae_cats = set()
            for col in ("art_forms", "genres"):
                ae_cats.update(to_tokens(r.iloc[0].get(col)))
            if ae_cats & set(cats):
                bridge_event = ae
                break

        shared = cats[0] if cats else "?"
        cn = f"cat:{shared}"

        if bridge_event is not None:
            Gc.add_node(bridge_event, kind="event")
            Gc.add_node(cn, kind="category")
            edges = [
                (sample_user, bridge_event),
                (bridge_event, cn),
                (cn, eid),
            ]
            nodes = {sample_user, bridge_event, cn, eid}
        else:
            Gc.add_node(cn, kind="category")
            edges = [(sample_user, cn), (cn, eid)]
            nodes = {sample_user, cn, eid}

        for u, v in edges:
            Gc.add_edge(u, v)
        path_node_sets.append(nodes)
        path_edge_sets.append(edges)

    pos_c = nx.spring_layout(Gc, seed=42, k=2.5)

    # Draw faded base
    nx.draw_networkx_edges(Gc, pos_c, edge_color=FADED, width=1, alpha=0.3, ax=ax, arrows=True)

    # Draw each path in its own colour
    for idx, (edges, nodes) in enumerate(zip(path_edge_sets, path_node_sets)):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        nx.draw_networkx_edges(
            Gc, pos_c, edgelist=edges,
            edge_color=color, width=3.0, alpha=0.85,
            arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax,
        )

    # Draw nodes
    for kind, color, size in [
        ("user", USER_COLOR, 900),
        ("event", EVENT_COLOR, 500),
        ("category", CATEGORY_COLOR, 500),
        ("recommended", RECO_COLOR, 700),
    ]:
        nl = [n for n, d in Gc.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(
            Gc, pos_c, nodelist=nl, node_color=color,
            node_size=size, edgecolors="white", linewidths=1.5, ax=ax,
        )

    lbl_c = {}
    for n in Gc.nodes():
        if n == sample_user:
            lbl_c[n] = f"YOU\n{n}"
        elif str(n).startswith("cat:"):
            lbl_c[n] = n.replace("cat:", "")
        else:
            lbl_c[n] = str(n)
    nx.draw_networkx_labels(Gc, pos_c, lbl_c, font_size=7, font_weight="bold", ax=ax)

    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Each colored path = one recommendation's evidence trail\n"
        + "Path: YOU attended an event → that event has a category → NEW event shares that category",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
    )
    ax.axis("off")

    # ---- MIDDLE: Similar-user paths ----
    ax = axes[1]
    ax.set_title(
        "Path Type B: Collaborative Filtering\n"
        + "YOU → shared event ← similar user → THEIR other events",
        fontsize=10, fontweight="bold",
    )

    Gs = nx.DiGraph()
    Gs.add_node(sample_user, kind="user")

    su_path_nodes: list[set] = []
    su_path_edges: list[list[tuple]] = []

    for idx, (eid, (sim, via_user)) in enumerate(sim_recs):
        Gs.add_node(eid, kind="recommended")
        Gs.add_node(via_user, kind="similar_user")

        via_attended = set(attends.loc[attends["user_id"] == via_user, "event_id"])
        shared_event = None
        for ae in attended_ids & via_attended:
            shared_event = ae
            break

        if shared_event is not None:
            Gs.add_node(shared_event, kind="event")
            edges = [
                (sample_user, shared_event),
                (via_user, shared_event),
                (via_user, eid),
            ]
            nodes = {sample_user, shared_event, via_user, eid}
        else:
            edges = [(sample_user, via_user), (via_user, eid)]
            nodes = {sample_user, via_user, eid}

        for u, v in edges:
            Gs.add_edge(u, v)
        su_path_nodes.append(nodes)
        su_path_edges.append(edges)

    pos_s = nx.spring_layout(Gs, seed=99, k=2.5)

    nx.draw_networkx_edges(Gs, pos_s, edge_color=FADED, width=1, alpha=0.3, ax=ax, arrows=True)

    for idx, (edges, nodes) in enumerate(zip(su_path_edges, su_path_nodes)):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        nx.draw_networkx_edges(
            Gs, pos_s, edgelist=edges,
            edge_color=color, width=3.0, alpha=0.85,
            arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax,
        )

    for kind, color, size in [
        ("user", USER_COLOR, 900),
        ("event", EVENT_COLOR, 500),
        ("similar_user", SIM_USER_COLOR, 700),
        ("recommended", RECO_COLOR, 700),
    ]:
        nl = [n for n, d in Gs.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(
            Gs, pos_s, nodelist=nl, node_color=color,
            node_size=size, edgecolors="white", linewidths=1.5, ax=ax,
        )

    lbl_s = {}
    for n in Gs.nodes():
        if n == sample_user:
            lbl_s[n] = f"YOU\n{n}"
        else:
            lbl_s[n] = str(n)
    nx.draw_networkx_labels(Gs, pos_s, lbl_s, font_size=7, font_weight="bold", ax=ax)

    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Each colored path = one recommendation's social proof\n"
        + "Path: YOU and a similar user both attended the same event → recommend THEIR other events",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
    )
    ax.axis("off")

    # ---- RIGHT PANEL: Artist paths ----
    ax = axes[2]
    ax.set_title(
        "Path Type C: Artist Discovery\n"
        + "YOU → interests/categories → artists + similar users → THEIR follows",
        fontsize=10, fontweight="bold",
    )

    Ga = nx.DiGraph()
    Ga.add_node(sample_user, kind="user")

    art_path_nodes: list[set] = []
    art_path_edges: list[list[tuple]] = []

    # Profile-matched artist paths
    for idx, (aid, (score, cats)) in enumerate(profile_artist_recs):
        Ga.add_node(aid, kind="artist")
        shared = cats[0] if cats else None
        if shared:
            cn = f"cat:{shared}"
            Ga.add_node(cn, kind="category")
            edges = [(sample_user, cn), (cn, aid)]
            nodes = {sample_user, cn, aid}
        else:
            edges = [(sample_user, aid)]
            nodes = {sample_user, aid}
        for u, v in edges:
            Ga.add_edge(u, v)
        art_path_nodes.append(nodes)
        art_path_edges.append(edges)

    # Collaborative artist paths
    for idx, (aid, (sim, via_user)) in enumerate(collab_artist_recs):
        if aid not in Ga:
            Ga.add_node(aid, kind="artist")
        Ga.add_node(via_user, kind="similar_user")

        via_attended = set(attends.loc[attends["user_id"] == via_user, "event_id"])
        shared_event = None
        for ae in attended_ids & via_attended:
            shared_event = ae
            break

        if shared_event is not None:
            Ga.add_node(shared_event, kind="event")
            edges = [(sample_user, shared_event), (via_user, shared_event), (via_user, aid)]
            nodes = {sample_user, shared_event, via_user, aid}
        else:
            edges = [(sample_user, via_user), (via_user, aid)]
            nodes = {sample_user, via_user, aid}

        for u, v in edges:
            Ga.add_edge(u, v)
        art_path_nodes.append(nodes)
        art_path_edges.append(edges)

    pos_a = nx.spring_layout(Ga, seed=55, k=2.5)

    nx.draw_networkx_edges(Ga, pos_a, edge_color=FADED, width=1, alpha=0.3, ax=ax, arrows=True)

    for idx, (edges, nodes) in enumerate(zip(art_path_edges, art_path_nodes)):
        color = PATH_COLORS[idx % len(PATH_COLORS)]
        nx.draw_networkx_edges(
            Ga, pos_a, edgelist=edges,
            edge_color=color, width=3.0, alpha=0.85,
            arrows=True, arrowstyle="-|>", arrowsize=15, ax=ax,
        )

    for kind, color, size in [
        ("user", USER_COLOR, 900),
        ("event", EVENT_COLOR, 500),
        ("category", CATEGORY_COLOR, 500),
        ("similar_user", SIM_USER_COLOR, 700),
        ("artist", ARTIST_COLOR, 700),
    ]:
        nl = [n for n, d in Ga.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(
            Ga, pos_a, nodelist=nl, node_color=color,
            node_size=size, edgecolors="white", linewidths=1.5, ax=ax,
        )

    lbl_a = {}
    for n in Ga.nodes():
        if n == sample_user:
            lbl_a[n] = f"YOU\n{n}"
        elif str(n).startswith("cat:"):
            lbl_a[n] = n.replace("cat:", "")
        else:
            lbl_a[n] = str(n)
    nx.draw_networkx_labels(Ga, pos_a, lbl_a, font_size=7, font_weight="bold", ax=ax)

    # Add explanation text box
    ax.text(
        0.5, -0.08,
        "Top paths: YOUR interests → matching artist categories (profile matching)\n"
        + "Bottom paths: similar users → artists THEY follow (collaborative discovery)",
        transform=ax.transAxes,
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="peachpuff", alpha=0.7),
    )
    ax.axis("off")

    # ── Legend ───────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=USER_COLOR, label="Target user (YOU)"),
        mpatches.Patch(color=EVENT_COLOR, label="Attended event"),
        mpatches.Patch(color=CATEGORY_COLOR, label="Category bridge"),
        mpatches.Patch(color=SIM_USER_COLOR, label="Similar user"),
        mpatches.Patch(color=RECO_COLOR, label="Recommended event"),
        mpatches.Patch(color=ARTIST_COLOR, label="Recommended artist"),
    ]
    for i, (eid, _) in enumerate(cat_recs):
        legend_handles.append(
            mpatches.Patch(color=PATH_COLORS[i], label=f"Path {i + 1}")
        )
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=5, fontsize=8, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    out_path = (ROOT / "reports" / "figures") / "5_paths.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n    Graph saved to:  {out_path}")

    plt.show()
    print_banner("End of Section 5")
    print()


if __name__ == "__main__":
    main()
