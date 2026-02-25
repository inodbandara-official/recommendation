"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Build the Basic Graph Structure
=============================================================

Run:  python pipeline/2_build_graph.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

DATA_DIR = Path("data")

# ── Colour palette ──────────────────────────────────────────
USER_COLOR = "#4A90D9"       # blue
EVENT_COLOR = "#E8A838"      # amber
CATEGORY_COLOR = "#50C878"   # green
ATTEND_COLOR = "#888888"     # grey
BELONGS_COLOR = "#CC6677"    # rose


def to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return {p.strip().strip("'\" ").lower() for p in s.split(",") if p.strip().strip("'\" ")}


def print_banner(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---\n")


def main() -> None:
    # ── Load data ───────────────────────────────────────────
    users = pd.read_csv(DATA_DIR / "users.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")

    # ── Build full graph ────────────────────────────────────
    G = nx.Graph()

    # Add user nodes
    for uid in users["user_id"]:
        G.add_node(uid, kind="user")

    # Add event nodes
    for eid in events["event_id"]:
        G.add_node(eid, kind="event")

    # Add category nodes and event→category edges
    for _, row in events.iterrows():
        eid = row["event_id"]
        cats = set()
        for col in ("art_forms", "genres"):
            if col in row and pd.notna(row[col]):
                cats.update(to_tokens(row[col]))
        for cat in cats:
            cat_node = f"cat:{cat}"
            if cat_node not in G:
                G.add_node(cat_node, kind="category")
            G.add_edge(eid, cat_node, relation="belongs_to")

    # Add user→event edges (attends)
    for _, row in attends.iterrows():
        uid = row["user_id"]
        eid = row["event_id"]
        if uid in G and eid in G:
            G.add_edge(uid, eid, relation="attended")

    # ── Full graph stats ────────────────────────────────────
    user_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "user"]
    event_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "event"]
    cat_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "category"]
    attend_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "attended"]
    belongs_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "belongs_to"]

    print_banner("2: Basic Graph Structure")

    print_section("Full Graph Summary")
    print(f"    User nodes          :  {len(user_nodes):,}")
    print(f"    Event nodes         :  {len(event_nodes):,}")
    print(f"    Category nodes      :  {len(cat_nodes):,}")
    print(f"    Total nodes         :  {G.number_of_nodes():,}")
    print()
    print(f"    Attended edges      :  {len(attend_edges):,}   (user ➜ event)")
    print(f"    Belongs-to edges    :  {len(belongs_edges):,}   (event ➜ category)")
    print(f"    Total edges         :  {G.number_of_edges():,}")

    # ── What each element represents ────────────────────────
    print_section("What Each Node Represents")
    print("    USER  (blue)   — A person registered in the system.")
    print("                     Has preferences like art interests,")
    print("                     city, mood, and activity level.")
    print()
    print("    EVENT (amber)  — A cultural event such as a concert,")
    print("                     exhibition, workshop, or competition.")
    print("                     Has genre, city, venue, and price.")
    print()
    print("    CATEGORY (green) — An art form or genre label such as")
    print("                     'music', 'dance', 'drama', 'film'.")
    print("                     Shared categories link events together.")

    print_section("What Each Edge Represents")
    print("    USER ──attended──➜ EVENT")
    print("        The user RSVP'd or went to that event.")
    print("        This is our primary interaction signal.")
    print()
    print("    EVENT ──belongs_to──➜ CATEGORY")
    print("        The event is tagged with that art form or genre.")
    print("        This lets us discover similar events through")
    print("        shared categories.")

    # ── Build a small subset for visualisation ──────────────
    # Let the user choose 3-5 user IDs
    user_counts = attends.groupby("user_id").size().sort_values(ascending=False)
    active = user_counts.head(20)
    print_section("Select Users for Graph Visualisation")
    print("    Active users (by attendance count):")
    for uid, cnt in active.items():
        uname = users.loc[users["user_id"] == uid, "name"].values
        uname = uname[0] if len(uname) else "?"
        print(f"      {uid}  ({cnt} events)  {uname}")
    print()
    raw = input("    Enter 3 to 5 user_ids (comma-separated): ").strip()
    entered = [u.strip() for u in raw.split(",") if u.strip()]
    valid_ids = set(users["user_id"])
    top_users = [u for u in entered if u in valid_ids]
    invalid = [u for u in entered if u not in valid_ids]
    if invalid:
        print(f"    Skipping unknown IDs: {invalid}")
    if len(top_users) < 3:
        # Pad with top active users not already chosen
        for uid in active.index:
            if uid not in top_users:
                top_users.append(uid)
            if len(top_users) >= 3:
                break
        print(f"    Not enough valid IDs — padded to: {top_users}")
    elif len(top_users) > 5:
        top_users = top_users[:5]
        print(f"    Trimmed to first 5: {top_users}")
    print(f"\n    Using {len(top_users)} users: {top_users}")

    # Collect their attended events (limit 4 per user for clarity)
    subset_edges_attend = []
    subset_events = set()
    for uid in top_users:
        user_events = attends.loc[attends["user_id"] == uid, "event_id"].head(4).tolist()
        for eid in user_events:
            subset_edges_attend.append((uid, eid))
            subset_events.add(eid)

    # Collect categories for those events (limit 2 per event)
    subset_edges_belongs = []
    subset_cats = set()
    for eid in subset_events:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        row = row.iloc[0]
        cats = set()
        for col in ("art_forms", "genres"):
            if col in row and pd.notna(row[col]):
                cats.update(to_tokens(row[col]))
        for cat in list(cats)[:2]:
            cat_node = f"cat:{cat}"
            subset_edges_belongs.append((eid, cat_node))
            subset_cats.add(cat_node)

    # Build subset graph
    S = nx.Graph()
    for uid in top_users:
        S.add_node(uid, kind="user")
    for eid in subset_events:
        S.add_node(eid, kind="event")
    for cat in subset_cats:
        S.add_node(cat, kind="category")
    for u, v in subset_edges_attend:
        S.add_edge(u, v, relation="attended")
    for u, v in subset_edges_belongs:
        S.add_edge(u, v, relation="belongs_to")

    print_section("Subset for Visualisation")
    print(f"    Users shown     :  {len(top_users)}  {top_users}")
    print(f"    Events shown    :  {len(subset_events)}")
    print(f"    Categories shown:  {len(subset_cats)}")
    print(f"    Total edges     :  {S.number_of_edges()}")

    # ── Draw ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_title(
        "Basic Graph: Users → Events → Categories",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )

    pos = nx.spring_layout(S, seed=42, k=1.8)

    # Separate edge lists for colouring
    attend_el = [(u, v) for u, v, d in S.edges(data=True) if d.get("relation") == "attended"]
    belongs_el = [(u, v) for u, v, d in S.edges(data=True) if d.get("relation") == "belongs_to"]

    nx.draw_networkx_edges(S, pos, edgelist=attend_el, edge_color=ATTEND_COLOR, width=1.5, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(S, pos, edgelist=belongs_el, edge_color=BELONGS_COLOR, style="dashed", width=1.2, alpha=0.6, ax=ax)

    # Draw nodes by kind
    for kind, color, size in [("user", USER_COLOR, 700), ("event", EVENT_COLOR, 500), ("category", CATEGORY_COLOR, 400)]:
        nodes = [n for n, d in S.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(S, pos, nodelist=nodes, node_color=color, node_size=size, edgecolors="white", linewidths=1.2, ax=ax)

    # Labels
    labels = {}
    for n, d in S.nodes(data=True):
        if d["kind"] == "category":
            labels[n] = n.replace("cat:", "")
        else:
            labels[n] = n
    nx.draw_networkx_labels(S, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Legend
    legend_handles = [
        mpatches.Patch(color=USER_COLOR, label="User node"),
        mpatches.Patch(color=EVENT_COLOR, label="Event node"),
        mpatches.Patch(color=CATEGORY_COLOR, label="Category node"),
        mpatches.Patch(color=ATTEND_COLOR, label="attended (user→event)"),
        mpatches.Patch(color=BELONGS_COLOR, label="belongs_to (event→category)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.axis("off")
    plt.tight_layout()

    out_path = Path("pipeline") / "2_graph.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n    Graph saved to:  {out_path}")

    plt.show()

    print_banner("End of Section 2")
    print()


if __name__ == "__main__":
    main()
