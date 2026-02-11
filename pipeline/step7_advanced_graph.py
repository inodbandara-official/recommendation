"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Step 7 — Advanced Graph View
=============================================================

Shows a comprehensive multi-path graph for the top-N
recommendations with full path traces, node importance
(degree centrality), and a summary scoreboard.

Run:  python pipeline/step7_advanced_graph.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# ── Palette ─────────────────────────────────────────────────
USER_COLOR = "#4A90D9"
EVENT_COLOR = "#E8A838"
CATEGORY_COLOR = "#50C878"
RECO_COLOR = "#E05555"
SIM_USER_COLOR = "#9B59B6"
PATH_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1",
    "#96CEB4", "#FFEAA7", "#DDA0DD",
    "#98D8C8", "#F7DC6F",
]
FADED = "#E8E8E8"


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
    users = pd.read_csv(DATA_DIR / "users.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    follows = pd.read_csv(DATA_DIR / "follows.csv")

    # ── Select a user ────────────────────────────────────────
    user_counts = attends.groupby("user_id").size().sort_values(ascending=False)
    print_banner("STEP 7: Advanced Graph View")

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
    attended_ids = set(attends.loc[attends["user_id"] == sample_user, "event_id"])

    print_section(f"Target User: {sample_user}")
    print(f"    Name           :  {user_row['name']}")
    print(f"    Art interests  :  {user_row.get('art_interests', 'N/A')}")
    print(f"    Region pref    :  {user_row.get('region_preference', 'N/A')}")
    print(f"    Attended       :  {len(attended_ids)} events")

    TOP_N = 8

    # ── Gather categories ────────────────────────────────────
    user_cats: set[str] = set()
    for eid in attended_ids:
        r = events.loc[events["event_id"] == eid]
        if r.empty:
            continue
        for col in ("art_forms", "genres"):
            user_cats.update(to_tokens(r.iloc[0].get(col)))

    # ── Category-path scoring ────────────────────────────────
    cat_scores: dict[str, tuple[float, list[str]]] = {}
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        if eid in attended_ids:
            continue
        ev_cats = set()
        for col in ("art_forms", "genres"):
            ev_cats.update(to_tokens(ev.get(col)))
        overlap = user_cats & ev_cats
        if overlap and user_cats:
            cat_scores[eid] = (len(overlap) / len(user_cats), sorted(overlap))

    # ── Similar-user scoring ─────────────────────────────────
    co_users = (
        attends.loc[
            attends["event_id"].isin(attended_ids)
            & (attends["user_id"] != sample_user)
        ]
        .groupby("user_id")["event_id"]
        .apply(set)
    )
    sim_map: dict[str, float] = {}
    for uid, their in co_users.items():
        j = len(their & attended_ids) / len(their | attended_ids)
        if j > 0:
            sim_map[uid] = j
    top_sim = sorted(sim_map.items(), key=lambda kv: kv[1], reverse=True)[:10]

    sim_event_scores: dict[str, tuple[float, str]] = {}
    for uid, sim in top_sim:
        their = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for eid in their - attended_ids:
            if eid not in sim_event_scores or sim > sim_event_scores[eid][0]:
                sim_event_scores[eid] = (sim, uid)

    # ── Combined score ───────────────────────────────────────
    all_candidates: set[str] = set(cat_scores.keys()) | set(sim_event_scores.keys())
    combined: dict[str, dict] = {}
    for eid in all_candidates:
        cat_s, cat_list = cat_scores.get(eid, (0.0, []))
        sim_s, via_user = sim_event_scores.get(eid, (0.0, ""))
        final = 0.5 * cat_s + 0.5 * sim_s
        combined[eid] = {
            "event_id": eid,
            "cat_score": cat_s,
            "sim_score": sim_s,
            "final_score": final,
            "categories": cat_list,
            "via_user": via_user,
        }

    ranked = sorted(combined.values(), key=lambda d: d["final_score"], reverse=True)[:TOP_N]

    # ── Print scoreboard ─────────────────────────────────────
    print_section(f"Top-{TOP_N} Recommendations (Combined)")
    print(f"    {'#':<3} {'Event':<8} {'Name':<38s} {'Cat':>5} {'Sim':>5} {'Final':>6}  Via")
    print("    " + "-" * 90)
    for i, rec in enumerate(ranked, 1):
        eid = rec["event_id"]
        ev_name = events.loc[events["event_id"] == eid, "name"]
        name = (ev_name.iloc[0] if not ev_name.empty else "?")[:37]
        via = rec["via_user"] if rec["via_user"] else "category"
        print(
            f"    {i:<3} {eid:<8} {name:<38s} "
            f"{rec['cat_score']:>5.2f} {rec['sim_score']:>5.3f} {rec['final_score']:>6.3f}  {via}"
        )

    # ── Build comprehensive graph ────────────────────────────
    G = nx.DiGraph()
    G.add_node(sample_user, kind="user")

    # Attended events (show top by relevance)
    shown_attended: list[str] = []
    for rec in ranked:
        for cat in rec["categories"][:2]:
            for ae in attended_ids:
                r = events.loc[events["event_id"] == ae]
                if r.empty:
                    continue
                ae_cats = set()
                for col in ("art_forms", "genres"):
                    ae_cats.update(to_tokens(r.iloc[0].get(col)))
                if cat in ae_cats and ae not in shown_attended:
                    shown_attended.append(ae)
                if len(shown_attended) >= 6:
                    break
            if len(shown_attended) >= 6:
                break
        if len(shown_attended) >= 6:
            break

    for eid in shown_attended:
        G.add_node(eid, kind="event")
        G.add_edge(sample_user, eid, relation="attended")

    # Categories
    shown_cats: set[str] = set()
    for rec in ranked:
        for cat in rec["categories"][:2]:
            shown_cats.add(cat)
    for eid in shown_attended:
        r = events.loc[events["event_id"] == eid]
        if r.empty:
            continue
        for col in ("art_forms", "genres"):
            for cat in to_tokens(r.iloc[0].get(col)):
                if cat in shown_cats:
                    shown_cats.add(cat)

    for cat in shown_cats:
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

    # Similar users
    shown_sim_users: set[str] = set()
    for rec in ranked:
        if rec["via_user"]:
            shown_sim_users.add(rec["via_user"])
    for uid in shown_sim_users:
        G.add_node(uid, kind="similar_user")
        their_attended = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for ae in shown_attended:
            if ae in their_attended:
                G.add_edge(uid, ae, relation="attended")

    # Recommended events
    path_info: list[dict] = []
    for idx, rec in enumerate(ranked):
        eid = rec["event_id"]
        G.add_node(eid, kind="recommended")

        edges_this: list[tuple] = []
        nodes_this: set = {sample_user, eid}

        # Category path
        for cat in rec["categories"][:2]:
            cn = f"cat:{cat}"
            if cn in G:
                G.add_edge(cn, eid, relation="recommends")
                edges_this.append((cn, eid))
                nodes_this.add(cn)
                # Find bridge event
                for ae in shown_attended:
                    if G.has_edge(ae, cn):
                        edges_this.append((sample_user, ae))
                        edges_this.append((ae, cn))
                        nodes_this.update({ae})
                        break

        # Similar-user path
        if rec["via_user"] and rec["via_user"] in G:
            uid = rec["via_user"]
            G.add_edge(uid, eid, relation="recommends")
            edges_this.append((uid, eid))
            nodes_this.add(uid)

        path_info.append({
            "idx": idx,
            "event_id": eid,
            "edges": edges_this,
            "nodes": nodes_this,
        })

    # ── Degree centrality ────────────────────────────────────
    centrality = nx.degree_centrality(G)

    print_section("Key Node Centralities")
    top_central = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:12]
    for node, cent in top_central:
        kind = G.nodes[node].get("kind", "?")
        label = str(node).replace("cat:", "CAT:")
        print(f"    {label:<25s}  {kind:<15s}  centrality={cent:.3f}")

    # ── Visualise ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(
        f"Step 7 — Advanced Graph View for {sample_user}",
        fontsize=15,
        fontweight="bold",
    )

    # ---- LEFT: Full multi-path graph ----
    ax = axes[0]
    ax.set_title("Multi-Path Recommendation Graph", fontsize=12, fontweight="bold")

    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=50)

    # Base edges (faded)
    all_edges = list(G.edges())
    nx.draw_networkx_edges(
        G, pos, edgelist=all_edges,
        edge_color=FADED, width=0.8, alpha=0.3, arrows=True, ax=ax,
    )

    # Colour each recommendation path differently
    for pinfo in path_info:
        color = PATH_COLORS[pinfo["idx"] % len(PATH_COLORS)]
        if pinfo["edges"]:
            nx.draw_networkx_edges(
                G, pos, edgelist=pinfo["edges"],
                edge_color=color, width=2.5, alpha=0.8,
                arrows=True, arrowstyle="-|>", arrowsize=12, ax=ax,
            )

    # Nodes sized by centrality
    for kind, base_color, base_size in [
        ("user", USER_COLOR, 1000),
        ("event", EVENT_COLOR, 400),
        ("category", CATEGORY_COLOR, 400),
        ("similar_user", SIM_USER_COLOR, 500),
        ("recommended", RECO_COLOR, 600),
    ]:
        nl = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not nl:
            continue
        sizes = [base_size + centrality.get(n, 0) * 2000 for n in nl]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nl, node_color=base_color,
            node_size=sizes, edgecolors="white", linewidths=1.5,
            alpha=0.9, ax=ax,
        )

    labels = {}
    for n in G.nodes():
        if n == sample_user:
            labels[n] = f"YOU\n{n}"
        elif str(n).startswith("cat:"):
            labels[n] = str(n).replace("cat:", "")
        else:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", ax=ax)
    ax.axis("off")

    # ---- RIGHT: Scoreboard + top paths ----
    ax2 = axes[1]
    ax2.set_title("Recommendation Scoreboard", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Draw table-style scoreboard
    y_start = 0.95
    line_h = 0.06
    x_positions = [0.02, 0.08, 0.42, 0.62, 0.75, 0.88]

    # Header
    headers = ["#", "Event", "Cat", "Sim", "Final", "Via"]
    for xi, header in zip(x_positions, headers):
        ax2.text(
            xi, y_start, header,
            transform=ax2.transAxes,
            fontsize=10, fontweight="bold", va="top",
            fontfamily="monospace",
        )

    y = y_start - line_h * 1.2
    ax2.plot([0.01, 0.99], [y + line_h * 0.3, y + line_h * 0.3], color="#ccc", linewidth=0.8, transform=ax2.transAxes, clip_on=False)

    for i, rec in enumerate(ranked):
        eid = rec["event_id"]
        ev_name = events.loc[events["event_id"] == eid, "name"]
        name = (ev_name.iloc[0] if not ev_name.empty else "?")[:30]
        via = rec["via_user"][:8] if rec["via_user"] else "category"
        color = PATH_COLORS[i % len(PATH_COLORS)]

        values = [
            f"{i + 1}.",
            f"{name}",
            f"{rec['cat_score']:.2f}",
            f"{rec['sim_score']:.3f}",
            f"{rec['final_score']:.3f}",
            via,
        ]
        for xi, val in zip(x_positions, values):
            ax2.text(
                xi, y, val,
                transform=ax2.transAxes,
                fontsize=8, va="top",
                fontfamily="monospace",
                color="#333",
            )
        # Colour indicator dot
        ax2.plot(
            0.01, y - line_h * 0.1, "o",
            color=color, markersize=8, alpha=0.7,
            transform=ax2.transAxes, clip_on=False,
        )
        y -= line_h

    # Path summary below scoreboard
    y -= line_h * 0.5
    ax2.text(
        0.02, y, "Path Traces:",
        transform=ax2.transAxes,
        fontsize=10, fontweight="bold", va="top",
    )
    y -= line_h * 0.8

    for i, rec in enumerate(ranked[:5]):
        eid = rec["event_id"]
        cats_str = ", ".join(rec["categories"][:2]) if rec["categories"] else "N/A"
        via = rec["via_user"] if rec["via_user"] else "N/A"
        color = PATH_COLORS[i % len(PATH_COLORS)]

        path_text = f"  {i + 1}. {eid}: via categories [{cats_str}]"
        if rec["via_user"]:
            path_text += f"  +  via user {via}"

        ax2.text(
            0.02, y, path_text,
            transform=ax2.transAxes,
            fontsize=7, va="top",
            fontfamily="monospace",
            color=color,
        )
        y -= line_h * 0.7

    # ── Legend ───────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=USER_COLOR, label="Target user"),
        mpatches.Patch(color=EVENT_COLOR, label="Attended event"),
        mpatches.Patch(color=CATEGORY_COLOR, label="Category"),
        mpatches.Patch(color=SIM_USER_COLOR, label="Similar user"),
        mpatches.Patch(color=RECO_COLOR, label="Recommended"),
    ]
    for i in range(min(TOP_N, len(PATH_COLORS))):
        legend_handles.append(mpatches.Patch(color=PATH_COLORS[i], label=f"Path {i + 1}"))

    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=7, fontsize=8, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])

    out_path = Path("pipeline") / "step7_advanced_graph.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n    Graph saved to:  {out_path}")

    plt.show()

    # ── Summary statistics ───────────────────────────────────
    print_section("Graph Summary Statistics")
    print(f"    Total nodes         :  {G.number_of_nodes()}")
    print(f"    Total edges         :  {G.number_of_edges()}")
    print(f"    Recommended events  :  {len(ranked)}")
    print(f"    Unique categories   :  {len(shown_cats)}")
    print(f"    Similar users shown :  {len(shown_sim_users)}")
    print(f"    Avg. centrality     :  {np.mean(list(centrality.values())):.4f}")
    print(f"    Max centrality node :  {top_central[0][0]}  ({top_central[0][1]:.3f})")

    print_banner("End of Step 7")
    print()


if __name__ == "__main__":
    main()
