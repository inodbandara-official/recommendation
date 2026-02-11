"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Step 6 — Dynamic User Input
=============================================================

Interactive mode: choose a specific user,
filter by categories, and see recommendations + graph
update in real time.

Run:  python pipeline/step6_dynamic_input.py
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
SIM_USER_COLOR = "#9B59B6"
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


def build_user_graph(
    user_id: str,
    users: pd.DataFrame,
    events: pd.DataFrame,
    attends: pd.DataFrame,
    category_filter: set[str] | None = None,
    top_n: int = 5,
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Build recommendations and a visual graph for a given user."""
    attended_ids = set(
        attends.loc[attends["user_id"] == user_id, "event_id"]
    )

    # Categories the user is linked to
    user_cats: set[str] = set()
    for eid in attended_ids:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        for col in ("art_forms", "genres"):
            user_cats.update(to_tokens(row.iloc[0].get(col)))

    # Apply category filter if provided
    if category_filter:
        user_cats = user_cats & category_filter

    # Score candidates
    scores: dict[str, tuple[float, list[str]]] = {}
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid in attended_ids:
            continue
        ev_cats = set()
        for col in ("art_forms", "genres"):
            ev_cats.update(to_tokens(ev.get(col)))
        if category_filter:
            ev_cats = ev_cats & category_filter
        overlap = user_cats & ev_cats
        if overlap and user_cats:
            scores[eid] = (len(overlap) / len(user_cats), sorted(overlap))

    recs = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]
    rec_df = pd.DataFrame(
        [
            {
                "event_id": eid,
                "event_name": events.loc[events["event_id"] == eid, "name"].iloc[0],
                "score": s,
                "shared_categories": ", ".join(cats),
            }
            for eid, (s, cats) in recs
        ]
    )

    # Build visualisation graph
    G = nx.DiGraph()
    G.add_node(user_id, kind="user")

    shown_attended = list(attended_ids)[:4]
    for eid in shown_attended:
        G.add_node(eid, kind="event")
        G.add_edge(user_id, eid, relation="attended")

    # Categories
    shown_cats: set[str] = set()
    for eid in shown_attended:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        for col in ("art_forms", "genres"):
            cats_here = to_tokens(row.iloc[0].get(col))
            if category_filter:
                cats_here = cats_here & category_filter
            shown_cats.update(cats_here)

    for cat in list(shown_cats)[:6]:
        cn = f"cat:{cat}"
        G.add_node(cn, kind="category")
        for eid in shown_attended:
            r = events.loc[events["event_id"] == eid]
            if r.empty:
                continue
            ev_cats = set()
            for col in ("art_forms", "genres"):
                ev_cats.update(to_tokens(r.iloc[0].get(col)))
            if cat in ev_cats:
                G.add_edge(eid, cn, relation="belongs_to")

    for eid, (s, cats) in recs[:top_n]:
        G.add_node(eid, kind="recommended")
        for cat in cats[:2]:
            cn = f"cat:{cat}"
            if cn in G:
                G.add_edge(cn, eid, relation="recommends")

    return rec_df, G


def draw_graph(G: nx.DiGraph, user_id: str, title: str, ax: plt.Axes) -> None:
    """Draw the recommendation graph on the given axes."""
    pos = nx.spring_layout(G, seed=42, k=2.2)

    # Faded base edges
    all_edges = list(G.edges())
    nx.draw_networkx_edges(
        G, pos, edgelist=all_edges,
        edge_color=FADED, width=1, alpha=0.3, arrows=True, ax=ax,
    )

    # Highlight recommendation edges
    reco_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation") == "recommends"
    ]
    normal_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation") != "recommends"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges,
        edge_color="#888", width=1.5, alpha=0.6, arrows=True, ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=reco_edges,
        edge_color=RECO_COLOR, width=2.5, alpha=0.8,
        style="dashed", arrows=True, arrowstyle="-|>", arrowsize=14, ax=ax,
    )

    # Nodes
    for kind, color, size in [
        ("user", USER_COLOR, 900),
        ("event", EVENT_COLOR, 500),
        ("category", CATEGORY_COLOR, 500),
        ("recommended", RECO_COLOR, 700),
    ]:
        nl = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nl, node_color=color,
            node_size=size, edgecolors="white", linewidths=1.5, ax=ax,
        )

    labels = {}
    for n in G.nodes():
        if n == user_id:
            labels[n] = f"YOU\n{n}"
        elif str(n).startswith("cat:"):
            labels[n] = str(n).replace("cat:", "")
        else:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")


def interactive_loop(
    users: pd.DataFrame,
    events: pd.DataFrame,
    attends: pd.DataFrame,
) -> None:
    """Main interactive loop for the recommendation system."""

    # Discover all categories
    all_cats: set[str] = set()
    for _, ev in events.iterrows():
        for col in ("art_forms", "genres"):
            all_cats.update(to_tokens(ev.get(col)))
    sorted_cats = sorted(all_cats)

    all_user_ids = sorted(users["user_id"].unique())

    while True:
        print_section("Choose an action")
        print("    1) Recommend for a specific user")
        print("    2) Recommend with category filter")
        print("    3) Compare two users side-by-side")
        print("    4) Show top active users")
        print("    q) Quit")
        choice = input("\n    Your choice: ").strip().lower()

        if choice == "q":
            print("\n    Goodbye!\n")
            break

        elif choice == "1":
            uid = input("    Enter user_id (or press Enter for random): ").strip()
            if not uid:
                uid = all_user_ids[len(all_user_ids) // 3]
            if uid not in set(all_user_ids):
                print(f"    User '{uid}' not found. Try again.")
                continue

            user_row = users.loc[users["user_id"] == uid].iloc[0]
            n_interactions = len(attends.loc[attends["user_id"] == uid])
            print(f"\n    User: {uid}  |  Name: {user_row['name']}  |  Interactions: {n_interactions}")

            rec_df, G = build_user_graph(uid, users, events, attends, top_n=8)
            if rec_df.empty:
                print("    No recommendations found for this user.")
                continue

            print_section("Recommendations")
            for i, (_, r) in enumerate(rec_df.iterrows(), 1):
                print(
                    f"    {i:>2}. {r['event_id']}  {r['event_name']:<38s}  "
                    f"score={r['score']:.2f}  [{r['shared_categories']}]"
                )

            fig, ax = plt.subplots(figsize=(12, 8))
            draw_graph(G, uid, f"Recommendations for {uid}", ax)
            legend_handles = [
                mpatches.Patch(color=USER_COLOR, label="You"),
                mpatches.Patch(color=EVENT_COLOR, label="Attended"),
                mpatches.Patch(color=CATEGORY_COLOR, label="Category"),
                mpatches.Patch(color=RECO_COLOR, label="Recommended"),
            ]
            fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9)
            plt.tight_layout(rect=[0, 0.06, 1, 0.96])

            out = Path("pipeline") / f"step6_{uid}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\n    Graph saved to:  {out}")
            plt.show()

        elif choice == "2":
            uid = input("    Enter user_id (or press Enter for random): ").strip()
            if not uid:
                uid = all_user_ids[len(all_user_ids) // 4]
            if uid not in set(all_user_ids):
                print(f"    User '{uid}' not found.")
                continue

            print(f"\n    Available categories ({len(sorted_cats)}):")
            for i, cat in enumerate(sorted_cats[:30], 1):
                print(f"      {i:>2}. {cat}")
            if len(sorted_cats) > 30:
                print(f"      ... and {len(sorted_cats) - 30} more")

            cat_input = input("\n    Enter category names (comma-separated): ").strip()
            cat_filter = {c.strip().lower() for c in cat_input.split(",") if c.strip()}
            if not cat_filter:
                print("    No categories entered. Skipping filter.")
                cat_filter = None

            rec_df, G = build_user_graph(
                uid, users, events, attends,
                category_filter=cat_filter, top_n=8,
            )
            if rec_df.empty:
                print("    No recommendations found with that filter.")
                continue

            filter_label = ", ".join(cat_filter) if cat_filter else "all"
            print_section(f"Recommendations (filter: {filter_label})")
            for i, (_, r) in enumerate(rec_df.iterrows(), 1):
                print(
                    f"    {i:>2}. {r['event_id']}  {r['event_name']:<38s}  "
                    f"score={r['score']:.2f}  [{r['shared_categories']}]"
                )

            fig, ax = plt.subplots(figsize=(12, 8))
            draw_graph(G, uid, f"Filtered Recommendations for {uid} [{filter_label}]", ax)
            plt.tight_layout()
            out = Path("pipeline") / f"step6_{uid}_filtered.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\n    Graph saved to:  {out}")
            plt.show()

        elif choice == "3":
            uid1 = input("    Enter first user_id: ").strip()
            uid2 = input("    Enter second user_id: ").strip()
            if uid1 not in set(all_user_ids) or uid2 not in set(all_user_ids):
                print("    One or both users not found.")
                continue

            rec1, G1 = build_user_graph(uid1, users, events, attends, top_n=5)
            rec2, G2 = build_user_graph(uid2, users, events, attends, top_n=5)

            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            fig.suptitle(
                f"Step 6 — Side-by-Side Comparison: {uid1} vs {uid2}",
                fontsize=14, fontweight="bold",
            )
            draw_graph(G1, uid1, f"User: {uid1}", axes[0])
            draw_graph(G2, uid2, f"User: {uid2}", axes[1])
            plt.tight_layout(rect=[0, 0, 1, 0.94])

            out = Path("pipeline") / f"step6_compare_{uid1}_{uid2}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"\n    Graph saved to:  {out}")
            plt.show()

            # Show overlap
            common = set(rec1["event_id"]) & set(rec2["event_id"])
            if common:
                print(f"\n    Common recommendations: {common}")
            else:
                print("\n    No overlapping recommendations.")

        elif choice == "4":
            top_active = (
                attends.groupby("user_id").size()
                .sort_values(ascending=False)
                .head(15)
            )
            print_section("Top 15 Most Active Users")
            for uid, count in top_active.items():
                name = users.loc[users["user_id"] == uid, "name"]
                name = name.iloc[0] if not name.empty else "?"
                print(f"    {uid}  {name:<30s}  {count} interactions")
            print("\n    (Pick any user_id above for option 1 or 2)")

        else:
            print("    Invalid choice. Try again.")


def main() -> None:
    users = pd.read_csv(DATA_DIR / "users.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")

    print_banner("STEP 6: Dynamic Panel Input")
    print("    This interactive mode lets you choose users,")
    print("    apply category filters, and compare recommendations")
    print("    in real time.\n")

    interactive_loop(users, events, attends)

    print_banner("End of Step 6")
    print()


if __name__ == "__main__":
    main()
