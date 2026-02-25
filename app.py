"""
=============================================================
  🎭 Cultural Event Recommendation System for Sri Lanka
  Streamlit Dashboard
=============================================================
  Run:  streamlit run app.py
=============================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

# ── Ensure src is importable ────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.knowledge_based import KnowledgeMatcher
from src.graph_based import recommend_from_similar_users, recommend_artists_from_similar_users
from src.trend_based import TrendWindowRecommender
from src.hybrid.hybrid_ranker import HybridRanker, DEFAULT_WEIGHTS
from src.hybrid.explanations import attach_explanations
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, coverage

# ── Constants ───────────────────────────────────────────────
DATA_DIR = ROOT / "data"
USER_COLOR = "#4A90D9"
EVENT_COLOR = "#E8A838"
CATEGORY_COLOR = "#50C878"
RECO_COLOR = "#E05555"
SIM_USER_COLOR = "#9B59B6"
ARTIST_COLOR = "#FF8C42"
PATH_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
FADED = "#D0D0D0"


# ── Helpers ─────────────────────────────────────────────────
def to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return {p.strip().strip("'\" ").lower() for p in s.split(",") if p.strip().strip("'\" ")}


def friendly_list(val: object) -> str:
    tokens = to_tokens(val)
    return ", ".join(sorted(tokens)) if tokens else "—"


def _render_event_interaction_graph(profiles, primary_uid, user_event_sets, events_df, user_colors_map):
    """Draw the User → Event interaction path graph."""
    primary_events = user_event_sets.get(primary_uid, set())

    all_compare_events = set()
    for p in profiles[1:]:
        all_compare_events |= user_event_sets.get(p["user_id"], set())
    common_all = primary_events & all_compare_events

    G_ev = nx.Graph()
    for idx, p in enumerate(profiles):
        uid = p["user_id"]
        label = f"⭐ {p['name']}" if uid == primary_uid else p["name"]
        G_ev.add_node(uid, kind="user", label=label)

    all_user_events = set()
    for p in profiles:
        all_user_events |= user_event_sets.get(p["user_id"], set())

    shared_ev_ids = set()
    for i, p1 in enumerate(profiles):
        for p2 in profiles[i + 1:]:
            shared_ev_ids |= (
                user_event_sets.get(p1["user_id"], set())
                & user_event_sets.get(p2["user_id"], set())
            )
    non_shared_ev = all_user_events - shared_ev_ids
    display_events = list(shared_ev_ids)[:12] + list(non_shared_ev)[
        : max(0, 15 - len(shared_ev_ids))
    ]

    for eid in display_events:
        ev_r = events_df.loc[events_df["event_id"] == eid]
        ev_name = ev_r.iloc[0]["name"][:20] if not ev_r.empty else str(eid)[:12]
        is_shared = eid in shared_ev_ids
        G_ev.add_node(eid, kind="shared_event" if is_shared else "event", label=ev_name)

    for p in profiles:
        uid = p["user_id"]
        for eid in display_events:
            if eid in user_event_sets.get(uid, set()):
                G_ev.add_edge(uid, eid, user=uid)

    if G_ev.number_of_edges() > 0:
        fig_ev, ax_ev = plt.subplots(figsize=(10, 6))
        pos_ev = nx.spring_layout(G_ev, seed=42, k=2.0, iterations=60)

        for idx, p in enumerate(profiles):
            uid = p["user_id"]
            edges = [(u, v) for u, v, d in G_ev.edges(data=True) if d.get("user") == uid]
            if edges:
                nx.draw_networkx_edges(
                    G_ev, pos_ev, edgelist=edges,
                    edge_color=user_colors_map[uid], width=1.8, alpha=0.5, ax=ax_ev,
                )

        user_nodes = [n for n, d in G_ev.nodes(data=True) if d.get("kind") == "user"]
        user_node_colors = [user_colors_map.get(n, USER_COLOR) for n in user_nodes]
        nx.draw_networkx_nodes(
            G_ev, pos_ev, nodelist=user_nodes, node_color=user_node_colors,
            node_size=800, edgecolors="white", linewidths=2, node_shape="o", ax=ax_ev,
        )

        shared_nodes = [n for n, d in G_ev.nodes(data=True) if d.get("kind") == "shared_event"]
        if shared_nodes:
            nx.draw_networkx_nodes(
                G_ev, pos_ev, nodelist=shared_nodes, node_color="#FFD700",
                node_size=500, edgecolors="#E8A838", linewidths=2, node_shape="s", ax=ax_ev,
            )

        other_ev_nodes = [n for n, d in G_ev.nodes(data=True) if d.get("kind") == "event"]
        if other_ev_nodes:
            nx.draw_networkx_nodes(
                G_ev, pos_ev, nodelist=other_ev_nodes, node_color=EVENT_COLOR,
                node_size=350, edgecolors="white", linewidths=1, node_shape="s", ax=ax_ev,
            )

        labels_ev = {n: d.get("label", str(n)[:10]) for n, d in G_ev.nodes(data=True)}
        nx.draw_networkx_labels(G_ev, pos_ev, labels_ev, font_size=7, font_weight="bold", ax=ax_ev)

        legend_handles = [
            mpatches.Patch(color=user_colors_map[p["user_id"]], label=p["name"])
            for p in profiles
        ] + [
            mpatches.Patch(color="#FFD700", label="Shared Event"),
            mpatches.Patch(color=EVENT_COLOR, label="Unique Event"),
        ]
        ax_ev.legend(handles=legend_handles, loc="upper left", fontsize=7, framealpha=0.9)
        ax_ev.set_title("User → Event Interaction Paths", fontsize=12, fontweight="bold")
        ax_ev.axis("off")
        plt.tight_layout()
        st.pyplot(fig_ev)
        plt.close(fig_ev)

        st.info(
            "**Reading this graph:** Each coloured circle is a user and each square is an event. "
            "Lines show which user attended which event — when two users' lines meet at the same "
            "event (gold squares), that's a **shared attendance**. More shared events = more similar tastes."
        )

    return common_all


def _render_artist_interaction_graph(profiles, primary_uid, user_artist_sets, artists_df, user_colors_map):
    """Draw the User → Artist interaction path graph."""
    primary_artists = user_artist_sets.get(primary_uid, set())

    all_compare_artists = set()
    for p in profiles[1:]:
        all_compare_artists |= user_artist_sets.get(p["user_id"], set())
    common_artists_all = primary_artists & all_compare_artists

    G_ar = nx.Graph()
    for idx, p in enumerate(profiles):
        uid = p["user_id"]
        label = f"⭐ {p['name']}" if uid == primary_uid else p["name"]
        G_ar.add_node(uid, kind="user", label=label)

    all_user_artists = set()
    for p in profiles:
        all_user_artists |= user_artist_sets.get(p["user_id"], set())

    shared_ar_ids = set()
    for i, p1 in enumerate(profiles):
        for p2 in profiles[i + 1:]:
            shared_ar_ids |= (
                user_artist_sets.get(p1["user_id"], set())
                & user_artist_sets.get(p2["user_id"], set())
            )
    non_shared_ar = all_user_artists - shared_ar_ids
    display_artists = list(shared_ar_ids)[:12] + list(non_shared_ar)[
        : max(0, 15 - len(shared_ar_ids))
    ]

    for aid in display_artists:
        a_r = artists_df.loc[artists_df["artist_id"] == aid]
        a_name = a_r.iloc[0]["name"][:20] if not a_r.empty else str(aid)[:12]
        is_shared = aid in shared_ar_ids
        G_ar.add_node(aid, kind="shared_artist" if is_shared else "artist", label=a_name)

    for p in profiles:
        uid = p["user_id"]
        for aid in display_artists:
            if aid in user_artist_sets.get(uid, set()):
                G_ar.add_edge(uid, aid, user=uid)

    if G_ar.number_of_edges() > 0:
        fig_ar, ax_ar = plt.subplots(figsize=(10, 6))
        pos_ar = nx.spring_layout(G_ar, seed=42, k=2.0, iterations=60)

        for idx, p in enumerate(profiles):
            uid = p["user_id"]
            edges = [(u, v) for u, v, d in G_ar.edges(data=True) if d.get("user") == uid]
            if edges:
                nx.draw_networkx_edges(
                    G_ar, pos_ar, edgelist=edges,
                    edge_color=user_colors_map[uid], width=1.8, alpha=0.5, ax=ax_ar,
                )

        user_nodes_ar = [n for n, d in G_ar.nodes(data=True) if d.get("kind") == "user"]
        user_node_colors_ar = [user_colors_map.get(n, USER_COLOR) for n in user_nodes_ar]
        nx.draw_networkx_nodes(
            G_ar, pos_ar, nodelist=user_nodes_ar, node_color=user_node_colors_ar,
            node_size=800, edgecolors="white", linewidths=2, node_shape="o", ax=ax_ar,
        )

        shared_a_nodes = [n for n, d in G_ar.nodes(data=True) if d.get("kind") == "shared_artist"]
        if shared_a_nodes:
            nx.draw_networkx_nodes(
                G_ar, pos_ar, nodelist=shared_a_nodes, node_color="#DDA0DD",
                node_size=500, edgecolors="#9B59B6", linewidths=2, node_shape="D", ax=ax_ar,
            )

        other_a_nodes = [n for n, d in G_ar.nodes(data=True) if d.get("kind") == "artist"]
        if other_a_nodes:
            nx.draw_networkx_nodes(
                G_ar, pos_ar, nodelist=other_a_nodes, node_color=ARTIST_COLOR,
                node_size=350, edgecolors="white", linewidths=1, node_shape="D", ax=ax_ar,
            )

        labels_ar = {n: d.get("label", str(n)[:10]) for n, d in G_ar.nodes(data=True)}
        nx.draw_networkx_labels(G_ar, pos_ar, labels_ar, font_size=7, font_weight="bold", ax=ax_ar)

        legend_handles_ar = [
            mpatches.Patch(color=user_colors_map[p["user_id"]], label=p["name"])
            for p in profiles
        ] + [
            mpatches.Patch(color="#DDA0DD", label="Shared Artist"),
            mpatches.Patch(color=ARTIST_COLOR, label="Unique Artist"),
        ]
        ax_ar.legend(handles=legend_handles_ar, loc="upper left", fontsize=7, framealpha=0.9)
        ax_ar.set_title("User → Artist Interaction Paths", fontsize=12, fontweight="bold")
        ax_ar.axis("off")
        plt.tight_layout()
        st.pyplot(fig_ar)
        plt.close(fig_ar)

        st.info(
            "**Reading this graph:** Each coloured circle is a user and each diamond is an artist. "
            "Lines show which user follows which artist — when two users' lines meet at the same "
            "artist (purple diamonds), that's a **shared follow**. More shared artists = more aligned cultural taste."
        )

    return common_artists_all


def render_user_comparison(
    primary_uid: str,
    compare_uids: list[str],
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    artists_df: pd.DataFrame,
    attends_df: pd.DataFrame,
    follows_df: pd.DataFrame,
    mode: str = "all",
):
    """Render a side-by-side comparison of user interactions.

    Args:
        mode: 'events' – only event-related sections,
              'artists' – only artist-related sections,
              'all' – everything.
    """
    show_events = mode in ("all", "events")
    show_artists = mode in ("all", "artists")
    all_uids = [primary_uid] + list(compare_uids)

    # ── Build per-user profiles ──────────────────────────
    profiles: list[dict] = []
    user_event_sets: dict[str, set] = {}
    user_artist_sets: dict[str, set] = {}
    for uid in all_uids:
        row = users_df.loc[users_df["user_id"] == uid]
        if row.empty:
            continue
        row = row.iloc[0]
        att_ids = set(attends_df.loc[attends_df["user_id"] == uid, "event_id"])
        fol_ids = set(follows_df.loc[follows_df["user_id"] == uid, "artist_id"])
        user_event_sets[uid] = att_ids
        user_artist_sets[uid] = fol_ids
        profiles.append({
            "user_id": uid,
            "name": row.get("name", "?"),
            "city": row.get("city", "—"),
            "art_interests": row.get("art_interests", ""),
            "culture_preferences": row.get("culture_preferences", ""),
            "events_attended": att_ids,
            "artists_followed": fol_ids,
        })

    if len(profiles) < 2:
        st.warning("Need at least one comparison user.")
        return

    primary = profiles[0]

    # ── 1. Summary table ────────────────────────────────
    st.markdown("#### Overview")
    summary_rows = []
    for p in profiles:
        is_primary = p["user_id"] == primary_uid
        row_data: dict = {
            "": "⭐ YOU" if is_primary else "",
            "User": p["user_id"],
            "Name": p["name"],
            "City": p["city"],
            "Art Interests": friendly_list(p["art_interests"]),
        }
        if show_events:
            row_data["Events Attended"] = len(p["events_attended"])
            row_data["Shared Events"] = (
                len(p["events_attended"] & primary["events_attended"]) if not is_primary else "—"
            )
        if show_artists:
            row_data["Artists Followed"] = len(p["artists_followed"])
            row_data["Shared Artists"] = (
                len(p["artists_followed"] & primary["artists_followed"]) if not is_primary else "—"
            )
        summary_rows.append(row_data)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Shared sets
    primary_events = user_event_sets.get(primary_uid, set())
    primary_artists = user_artist_sets.get(primary_uid, set())

    # Build user_colors_map for graph visualizations
    user_colors_map = {}
    for idx, p in enumerate(profiles):
        user_colors_map[p["user_id"]] = PATH_COLORS[idx % len(PATH_COLORS)]

    # ── 2. Event section ────────────────────────────────
    if show_events:
        st.markdown("#### 🎪 Event Attendance Comparison")

        ev_rows = []
        for p in profiles[1:]:
            comp_events = user_event_sets.get(p["user_id"], set())
            shared = primary_events & comp_events
            only_primary = primary_events - comp_events
            only_comp = comp_events - primary_events
            ev_rows.append({
                "Compared With": f"{p['name']} ({p['user_id']})",
                "Events in Common": len(shared),
                f"Only {primary['name']}": len(only_primary),
                f"Only {p['name']}": len(only_comp),
                "Jaccard Overlap": f"{len(shared)/max(len(primary_events | comp_events),1):.3f}",
            })
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        # Event Interaction Path Visualization
        st.markdown("##### 🗺️ Event Interaction Map")
        st.markdown(
            "This graph shows how each user is connected to the events they attended. "
            "**Shared events** (attended by more than one user) are highlighted — "
            "lines converging on the same event node reveal common interests."
        )

        common_all = _render_event_interaction_graph(
            profiles, primary_uid, user_event_sets, events_df, user_colors_map,
        )

        if common_all:
            with st.expander(f"Show {len(common_all)} shared events", expanded=False):
                shared_ev_rows = []
                for eid in sorted(common_all):
                    ev_r = events_df.loc[events_df["event_id"] == eid]
                    ev_name = ev_r.iloc[0]["name"] if not ev_r.empty else str(eid)
                    attended_by = [p["name"] for p in profiles if eid in user_event_sets.get(p["user_id"], set())]
                    shared_ev_rows.append({
                        "Event": ev_name,
                        "Event ID": eid,
                        "Attended By": ", ".join(attended_by),
                    })
                st.dataframe(pd.DataFrame(shared_ev_rows), use_container_width=True, hide_index=True)

        all_compare_events = set()
        for p in profiles[1:]:
            all_compare_events |= user_event_sets.get(p["user_id"], set())
        unique_to_others = all_compare_events - primary_events
        if unique_to_others:
            with st.expander(f"🔍 {len(unique_to_others)} events you haven't attended (from compared users)", expanded=False):
                disc_rows = []
                for eid in sorted(unique_to_others):
                    ev_r = events_df.loc[events_df["event_id"] == eid]
                    if ev_r.empty:
                        continue
                    ev = ev_r.iloc[0]
                    who = [p["name"] for p in profiles[1:] if eid in user_event_sets.get(p["user_id"], set())]
                    disc_rows.append({
                        "Event": ev["name"],
                        "City": ev.get("city", "—"),
                        "Art Forms": friendly_list(ev.get("art_forms")),
                        "Attended By": ", ".join(who),
                    })
                st.dataframe(pd.DataFrame(disc_rows), use_container_width=True, hide_index=True)

    # ── 3. Artist section ───────────────────────────────
    if show_artists:
        st.markdown("#### 🎨 Artist Follow Comparison")

        ar_rows = []
        for p in profiles[1:]:
            comp_artists = user_artist_sets.get(p["user_id"], set())
            shared = primary_artists & comp_artists
            only_primary = primary_artists - comp_artists
            only_comp = comp_artists - primary_artists
            ar_rows.append({
                "Compared With": f"{p['name']} ({p['user_id']})",
                "Artists in Common": len(shared),
                f"Only {primary['name']}": len(only_primary),
                f"Only {p['name']}": len(only_comp),
                "Jaccard Overlap": f"{len(shared)/max(len(primary_artists | comp_artists),1):.3f}",
            })
        st.dataframe(pd.DataFrame(ar_rows), use_container_width=True, hide_index=True)

        # Artist Interaction Path Visualization
        st.markdown("##### 🗺️ Artist Interaction Map")
        st.markdown(
            "This graph shows how each user is connected to the artists they follow. "
            "**Shared artists** (followed by more than one user) are highlighted — "
            "converging lines reveal artists with cross-user appeal."
        )

        common_artists_all = _render_artist_interaction_graph(
            profiles, primary_uid, user_artist_sets, artists_df, user_colors_map,
        )

        if common_artists_all:
            with st.expander(f"Show {len(common_artists_all)} shared followed artists", expanded=False):
                shared_ar_rows = []
                for aid in sorted(common_artists_all):
                    a_r = artists_df.loc[artists_df["artist_id"] == aid]
                    a_name = a_r.iloc[0]["name"] if not a_r.empty else str(aid)
                    followed_by = [p["name"] for p in profiles if aid in user_artist_sets.get(p["user_id"], set())]
                    shared_ar_rows.append({
                        "Artist": a_name,
                        "Artist ID": aid,
                        "Followed By": ", ".join(followed_by),
                    })
                st.dataframe(pd.DataFrame(shared_ar_rows), use_container_width=True, hide_index=True)

        all_compare_artists = set()
        for p in profiles[1:]:
            all_compare_artists |= user_artist_sets.get(p["user_id"], set())
        unique_artists_others = all_compare_artists - primary_artists
        if unique_artists_others:
            with st.expander(f"🔍 {len(unique_artists_others)} artists you don't follow (from compared users)", expanded=False):
                disc_ar_rows = []
                for aid in sorted(unique_artists_others):
                    a_r = artists_df.loc[artists_df["artist_id"] == aid]
                    if a_r.empty:
                        continue
                    a = a_r.iloc[0]
                    who = [p["name"] for p in profiles[1:] if aid in user_artist_sets.get(p["user_id"], set())]
                    disc_ar_rows.append({
                        "Artist": a["name"],
                        "City": a.get("city", "—"),
                        "Art Forms": friendly_list(a.get("art_forms")),
                        "Popularity": a.get("popularity", "—"),
                        "Followed By": ", ".join(who),
                    })
                st.dataframe(pd.DataFrame(disc_ar_rows), use_container_width=True, hide_index=True)

    # ── 4. Interest overlap heatmap ─────────────────────
    st.markdown("#### 🎯 Interest Overlap Matrix")
    interest_sets = {p["user_id"]: to_tokens(p["art_interests"]) for p in profiles}
    labels = [f"{p['name']}" for p in profiles]
    n = len(profiles)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si = interest_sets[profiles[i]["user_id"]]
            sj = interest_sets[profiles[j]["user_id"]]
            matrix[i][j] = len(si & sj) / max(len(si | sj), 1)

    fig_h, ax_h = plt.subplots(figsize=(6, 4))
    im = ax_h.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax_h.set_xticks(range(n)); ax_h.set_yticks(range(n))
    ax_h.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_h.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax_h.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8,
                      color="white" if matrix[i][j] > 0.5 else "black")
    fig_h.colorbar(im, ax=ax_h, label="Jaccard Similarity")
    ax_h.set_title("Art Interest Overlap (Jaccard)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_h)
    plt.close(fig_h)


# ── Data Loading (cached) ──────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    users = pd.read_csv(DATA_DIR / "users.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    artists = pd.read_csv(DATA_DIR / "artists.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    follows = pd.read_csv(DATA_DIR / "follows.csv")
    attends["timestamp"] = pd.to_datetime(attends["timestamp"])
    return users, events, artists, attends, follows


# ── Model runners (cached per user) ────────────────────────
@st.cache_data(show_spinner="Running recommendation models...")
def run_recommendation_pipeline(
    user_id: str,
    _users_hash: str,
    top_n: int = 10,
):
    users, events, artists, attends, follows = load_data()
    attends_sorted = attends.sort_values("timestamp")

    # Knowledge-based
    km = KnowledgeMatcher(budget_col=None)
    km.fit(users, events)
    knowledge_df = km.recommend(user_id, top_n=len(events))
    knowledge_scores = knowledge_df[["event_id", "KnowledgeScore"]]

    # Graph-based
    graph_df = recommend_from_similar_users(
        attends=attends_sorted, follows=follows,
        target_user=user_id, top_users=50, top_n=top_n * 3, alpha=0.5,
    )
    graph_scores = (
        graph_df[["event_id", "GraphScore"]]
        if not graph_df.empty
        else pd.DataFrame(columns=["event_id", "GraphScore"])
    )

    # Trend-based
    trend_model = TrendWindowRecommender().fit(attends_sorted)
    trend_df = trend_model.recommend(top_n=top_n * 3, window_days=14)
    trend_scores = trend_df[["event_id", "TrendScore"]]

    # Merge
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

    user_interactions = len(attends_sorted.loc[attends_sorted["user_id"] == user_id])
    ranker = HybridRanker()
    ranked = ranker.rank(merged, user_interactions=user_interactions, top_n=top_n)

    strategy = "cold_start" if user_interactions < ranker.interaction_threshold else "active"

    user_row = users.loc[users["user_id"] == user_id].iloc[0]
    interests = to_tokens(user_row.get("art_interests"))
    city = user_row.get("city") if pd.notna(user_row.get("city")) else None
    ranked = attach_explanations(ranked, events=events, user_interests=interests, user_city=city)

    return ranked, knowledge_scores, graph_scores, trend_scores, strategy, user_interactions


@st.cache_data(show_spinner="Computing artist recommendations...")
def run_artist_pipeline(user_id: str, _users_hash: str, top_n: int = 10):
    users, events, artists, attends, follows = load_data()
    user_row = users.loc[users["user_id"] == user_id].iloc[0]
    user_interests = to_tokens(user_row.get("art_interests", ""))
    followed_ids = set(follows.loc[follows["user_id"] == user_id, "artist_id"])

    # Profile-matched
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

    profile_recs = sorted(artist_profile.items(), key=lambda kv: kv[1][0], reverse=True)[:top_n]

    # ── Find similar users (Jaccard + Adamic-Adar) ──────────
    from src.graph_based.graph_similarity import (
        jaccard_similar_users, adamic_adar_similar_users, merge_similarity,
    )
    j_scores = jaccard_similar_users(attends, follows, user_id)
    aa_scores = adamic_adar_similar_users(attends, user_id)
    merged_sim = merge_similarity(j_scores, aa_scores, alpha=0.5)
    top_similar = sorted(merged_sim.items(), key=lambda kv: kv[1], reverse=True)[:5]

    # Build rich similar-user profiles
    similar_users_data: list[dict] = []
    for sim_uid, sim_score in top_similar:
        sim_row = users.loc[users["user_id"] == sim_uid]
        if sim_row.empty:
            continue
        sim_row = sim_row.iloc[0]
        sim_attended = set(attends.loc[attends["user_id"] == sim_uid, "event_id"])
        sim_followed = set(follows.loc[follows["user_id"] == sim_uid, "artist_id"])
        sim_interests = to_tokens(sim_row.get("art_interests", ""))
        shared_interests = user_interests & sim_interests
        shared_events = set(attends.loc[attends["user_id"] == user_id, "event_id"]) & sim_attended
        shared_artists = followed_ids & sim_followed

        similar_users_data.append({
            "user_id": sim_uid,
            "name": sim_row.get("name", "?"),
            "city": sim_row.get("city", "—"),
            "art_interests": sim_row.get("art_interests", ""),
            "similarity": sim_score,
            "jaccard": j_scores.get(sim_uid, 0.0),
            "adamic_adar": aa_scores.get(sim_uid, 0.0),
            "shared_interests": sorted(shared_interests),
            "shared_events": sorted(shared_events),
            "shared_artists": sorted(shared_artists),
            "events_attended": sorted(sim_attended),
            "artists_followed": sorted(sim_followed),
        })

    # Collaborative
    collab_df = recommend_artists_from_similar_users(
        attends=attends, follows=follows,
        target_user=user_id, top_users=50, top_n=top_n, alpha=0.5,
    )

    # Track which similar users contributed to each artist rec
    sim_map = dict(top_similar)
    artist_sources: dict[str, list[dict]] = {}  # artist_id -> [{user_id, name, sim}]
    for sim_info in similar_users_data:
        sid = sim_info["user_id"]
        for aid in sim_info["artists_followed"]:
            if aid not in followed_ids:  # only artists the target hasn't followed
                artist_sources.setdefault(aid, []).append({
                    "user_id": sid,
                    "name": sim_info["name"],
                    "similarity": sim_info["similarity"],
                })

    # Merge
    artist_final: dict[str, dict] = {}
    for aid, (score, cats) in profile_recs:
        artist_final[aid] = {"artist_id": aid, "ProfileScore": score, "CollabScore": 0.0,
                             "shared": cats, "sources": artist_sources.get(aid, [])}
    for _, row in collab_df.iterrows():
        aid = row["artist_id"]
        if aid in artist_final:
            artist_final[aid]["CollabScore"] = row["ArtistGraphScore"]
            if not artist_final[aid]["sources"]:
                artist_final[aid]["sources"] = artist_sources.get(aid, [])
        else:
            artist_final[aid] = {"artist_id": aid, "ProfileScore": 0.0,
                                 "CollabScore": row["ArtistGraphScore"], "shared": [],
                                 "sources": artist_sources.get(aid, [])}

    for d in artist_final.values():
        d["FinalArtistScore"] = 0.5 * d["ProfileScore"] + 0.5 * d["CollabScore"]

    artist_ranked = sorted(artist_final.values(), key=lambda d: d["FinalArtistScore"], reverse=True)[:top_n]
    return artist_ranked, similar_users_data


# ── Graph builders ──────────────────────────────────────────
def build_category_path_graph(user_id, events, attends):
    attended_ids = set(attends.loc[attends["user_id"] == user_id, "event_id"])
    user_cats: set[str] = set()
    for eid in attended_ids:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        for col in ("art_forms", "genres"):
            user_cats.update(to_tokens(row.iloc[0].get(col)))

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

    cat_recs = sorted(cat_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:5]

    G = nx.DiGraph()
    G.add_node(user_id, kind="user")
    shown_attended = list(attended_ids)[:3]
    for eid in shown_attended:
        G.add_node(eid, kind="event")
        G.add_edge(user_id, eid, relation="attended")

    shown_cats: set[str] = set()
    for eid in shown_attended:
        row = events.loc[events["event_id"] == eid]
        if row.empty:
            continue
        for col in ("art_forms", "genres"):
            for cat in to_tokens(row.iloc[0].get(col)):
                if cat in user_cats:
                    shown_cats.add(cat)

    for cat in list(shown_cats)[:4]:
        cn = f"cat:{cat}"
        G.add_node(cn, kind="category")
        for eid in shown_attended:
            row = events.loc[events["event_id"] == eid]
            if row.empty:
                continue
            ev_cats = set()
            for col in ("art_forms", "genres"):
                ev_cats.update(to_tokens(row.iloc[0].get(col)))
            if cat in ev_cats:
                G.add_edge(eid, cn, relation="belongs_to")

    for eid, (score, cats) in cat_recs[:3]:
        G.add_node(eid, kind="recommended")
        for cat in cats[:2]:
            cn = f"cat:{cat}"
            if cn in G:
                G.add_edge(cn, eid, relation="recommends")

    return G, cat_recs


def build_collab_graph(user_id, events, attends):
    attended_ids = set(attends.loc[attends["user_id"] == user_id, "event_id"])
    co_users = (
        attends.loc[attends["event_id"].isin(attended_ids) & (attends["user_id"] != user_id)]
        .groupby("user_id")["event_id"].apply(set)
    )
    sim_scores: dict[str, float] = {}
    for uid, their in co_users.items():
        jac = len(attended_ids & their) / max(len(attended_ids | their), 1)
        sim_scores[uid] = jac
    top_sim = sorted(sim_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]

    sim_event_scores: dict[str, tuple[float, str]] = {}
    for uid, sim in top_sim:
        their_events = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for eid in their_events - attended_ids:
            if eid not in sim_event_scores or sim > sim_event_scores[eid][0]:
                sim_event_scores[eid] = (sim, uid)
    sim_recs = sorted(sim_event_scores.items(), key=lambda kv: kv[1][0], reverse=True)[:5]

    G = nx.DiGraph()
    G.add_node(user_id, kind="user")
    shared_events = list(attended_ids)[:2]
    for eid in shared_events:
        G.add_node(eid, kind="event")
        G.add_edge(user_id, eid, relation="attended")

    for uid, sim in top_sim[:2]:
        G.add_node(uid, kind="similar_user")
        their_events = set(attends.loc[attends["user_id"] == uid, "event_id"])
        for eid in shared_events:
            if eid in their_events:
                G.add_edge(uid, eid, relation="attended")

    for eid, (sim, via) in sim_recs[:3]:
        G.add_node(eid, kind="recommended")
        if via in G:
            G.add_edge(via, eid, relation="recommends")

    return G, sim_recs


def draw_graph_on_ax(G, user_id, ax, title):
    pos = nx.spring_layout(G, seed=42, k=2.2)

    all_edges = list(G.edges())
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color=FADED, width=1, alpha=0.3, arrows=True, ax=ax)

    reco_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "recommends"]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") != "recommends"]

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#888", width=1.5, alpha=0.6, arrows=True, ax=ax)
    if reco_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=reco_edges, edge_color=RECO_COLOR, width=2.5, alpha=0.8,
            style="dashed", arrows=True, arrowstyle="-|>", arrowsize=14, ax=ax,
        )

    for kind, color, size in [
        ("user", USER_COLOR, 900), ("event", EVENT_COLOR, 500),
        ("category", CATEGORY_COLOR, 400), ("similar_user", SIM_USER_COLOR, 600),
        ("recommended", RECO_COLOR, 700), ("artist", ARTIST_COLOR, 600),
    ]:
        nl = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if nl:
            nx.draw_networkx_nodes(G, pos, nodelist=nl, node_color=color, node_size=size, edgecolors="white", linewidths=1.5, ax=ax)

    labels = {}
    for n in G.nodes():
        if n == user_id:
            labels[n] = f"YOU\n{n}"
        elif str(n).startswith("cat:"):
            labels[n] = str(n).replace("cat:", "")
        else:
            labels[n] = str(n)
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


# ── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Cultural Event Recommender for Rasaswadaya",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; font-size: 1.05rem; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0 0 0.3rem 0; font-size: 0.85rem; color: #888; text-transform: uppercase; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #333; }
    .explanation-chip {
        display: inline-block;
        background: #e8f4f8;
        color: #1a6f93;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        margin: 2px 3px;
    }
    .score-bar {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        position: relative;
        overflow: hidden;
    }
    .strategy-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .strategy-cold { background: #dbeafe; color: #1e40af; }
    .strategy-active { background: #d1fae5; color: #065f46; }
    div[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    /* Styled sidebar nav buttons */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        text-align: left !important;
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 10px 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        background: white !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%) !important;
        border-color: #667eea !important;
        transform: translateX(3px);
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ───────────────────────────────────────────────
users, events, artists, attends, follows = load_data()

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎭 Cultural Event Recommender for Rasaswadaya</h1>
    <p>Hybrid recommendation system blending Knowledge, Graph and Trend models
       for personalized cultural event &amp; artist discovery</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")

    _NAV_PAGES = [
        ("🏠", "Dashboard",             "🏠 Dashboard"),
        ("🎯", "Event Recommendations",  "🎯 Event Recommendations"),
        ("🎨", "Artist Discovery",       "🎨 Artist Discovery"),
        ("📊", "Model Comparison",       "📊 Model Comparison"),
        ("🔬", "Evaluation Metrics",     "🔬 Evaluation Metrics"),
        ("📈", "Data Explorer",          "📈 Data Explorer"),
    ]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "🏠 Dashboard"

    for icon, label, key in _NAV_PAGES:
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    page = st.session_state.active_page

    st.markdown("---")
    st.markdown("### 👤 Select User")

    user_counts = attends.groupby("user_id").size().sort_values(ascending=False)
    user_options = []
    for uid in users["user_id"]:
        name = users.loc[users["user_id"] == uid, "name"].iloc[0]
        count = user_counts.get(uid, 0)
        user_options.append(f"{uid} — {name} ({count} events)")

    selected_idx = st.selectbox(
        "Choose a user",
        range(len(user_options)),
        format_func=lambda i: user_options[i],
        index=0,
    )
    selected_user = users["user_id"].iloc[selected_idx]

    st.markdown("---")
    top_n = st.slider("Top-N recommendations", 5, 20, 10)

    st.markdown("---")
    st.markdown("### 📦 Dataset Stats")
    col1, col2 = st.columns(2)
    col1.metric("Users", f"{len(users):,}")
    col2.metric("Events", f"{len(events):,}")
    col1.metric("Artists", f"{len(artists):,}")
    col2.metric("Attends", f"{len(attends):,}")
    col1.metric("Follows", f"{len(follows):,}")

# ── User info helper ────────────────────────────────────────
user_row = users.loc[users["user_id"] == selected_user].iloc[0]
_cache_key = f"{selected_user}_{top_n}"

# =====================================================
# PAGE: Dashboard
# =====================================================
if page == "🏠 Dashboard":
    st.markdown("## 🏠 System Overview")

    # User profile card
    col_profile, col_stats = st.columns([1, 2])

    with col_profile:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Name:** {user_row['name']}")
        st.markdown(f"**City:** {user_row.get('city', '—')}")
        st.markdown(f"**Art Interests:** {friendly_list(user_row.get('art_interests'))}")
        st.markdown(f"**Culture Preferences:** {friendly_list(user_row.get('culture_preferences'))}")
        st.markdown(f"**Mood Preferences:** {friendly_list(user_row.get('mood_preferences'))}")
        st.markdown(f"**Activity Level:** {user_row.get('activity_level', '—')}")

        n_attended = len(attends.loc[attends["user_id"] == selected_user])
        n_follows = len(follows.loc[follows["user_id"] == selected_user])
        st.markdown(f"**Events Attended:** {n_attended}")
        st.markdown(f"**Artists Followed:** {n_follows}")

    with col_stats:
        st.markdown("### 🏗️ System Architecture")
        st.markdown("""
        This hybrid recommendation system combines **three independent models** to generate
        personalized event and artist recommendations for cultural events across Sri Lanka.

        | Model | Method | Signal |
        |-------|--------|--------|
        | **Knowledge-Based** | Profile matching (interests, city) → event attributes | Content overlap |
        | **Graph-Based** | User similarity via shared attendance (Jaccard + Adamic-Adar) | Collaborative filtering |
        | **Trend-Based** | Windowed attendance counts + growth rate | Temporal popularity |

        The **Hybrid Ranker** blends scores using dynamic weights based on user activity:
        - **Cold-start** (< 5 interactions): α=0.5 Knowledge, β=0.2 Graph, γ=0.3 Trend
        - **Active** (≥ 5 interactions): α=0.2 Knowledge, β=0.5 Graph, γ=0.3 Trend
        """)

    st.markdown("---")

    # Quick recommendations
    st.markdown("### ⚡ Quick Event Recommendations")
    st.markdown("""
    These are the **top 5 events** the system thinks this user would enjoy most.
    Each event is scored by blending three models — one that matches the user's
    interests, one that looks at what similar users attended, and one that tracks
    what's trending right now. The strategy badge below shows which blending
    approach was used for this user.
    """)
    ranked, k_scores, g_scores, t_scores, strategy, n_interactions = run_recommendation_pipeline(
        selected_user, _cache_key, top_n=5,
    )

    badge_class = "strategy-cold" if strategy == "cold_start" else "strategy-active"
    st.markdown(
        f'Strategy: <span class="strategy-badge {badge_class}">{strategy.replace("_", " ").title()}</span>'
        f' &nbsp; ({n_interactions} interactions)',
        unsafe_allow_html=True,
    )

    for _, r in ranked.iterrows():
        ev = events.loc[events["event_id"] == r["event_id"]]
        if ev.empty:
            continue
        ev = ev.iloc[0]
        with st.container():
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.markdown(f"**{ev['name']}** — {ev.get('city', '—')}")
                explanations = r.get("Explanations", [])
                if isinstance(explanations, list):
                    chips = " ".join(f'<span class="explanation-chip">{e}</span>' for e in explanations)
                    st.markdown(chips, unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Score:** {r['FinalScore']:.3f}")
            with c3:
                st.markdown(f"🎭 {friendly_list(ev.get('art_forms'))}")

    # Quick artist recommendations
    st.markdown("---")
    st.markdown("### 🎨 Quick Artist Recommendations")
    st.markdown("""
    These are the **top 5 artists** the system recommends for this user.
    Artists are scored using two signals: how well their art forms match the
    user's interests (**Profile Score**), and how many similar users already
    follow them (**Collab Score**). The final rank is a 50/50 blend of both.
    """)
    artist_recs_dash, _ = run_artist_pipeline(selected_user, _cache_key, top_n=5)

    if artist_recs_dash:
        for rec in artist_recs_dash:
            aid = rec["artist_id"]
            art_r = artists.loc[artists["artist_id"] == aid]
            if art_r.empty:
                continue
            art = art_r.iloc[0]
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{art['name']}** — {art.get('city', '—')}")
                    shared = rec.get("shared", [])
                    if shared:
                        chips = " ".join(f'<span class="explanation-chip">{s}</span>' for s in shared)
                        st.markdown(chips, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"**Score:** {rec['FinalArtistScore']:.3f}")
                with c3:
                    st.markdown(f"🎭 {friendly_list(art.get('art_forms'))}")
    else:
        st.caption("No artist recommendations available for this user.")


# =====================================================
# PAGE: Event Recommendations
# =====================================================
elif page == "🎯 Event Recommendations":
    st.markdown("## 🎯 Event Recommendations")
    st.markdown(f"Personalized event recommendations for **{user_row['name']}** from **{user_row.get('city', '—')}**")

    ranked, k_scores, g_scores, t_scores, strategy, n_interactions = run_recommendation_pipeline(
        selected_user, _cache_key, top_n=top_n,
    )

    # Strategy info
    badge_class = "strategy-cold" if strategy == "cold_start" else "strategy-active"
    scheme = DEFAULT_WEIGHTS[strategy]
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(
        f'<div class="metric-card"><h3>Strategy</h3><div class="value">{strategy.replace("_"," ").title()}</div></div>',
        unsafe_allow_html=True,
    )
    col_b.markdown(
        f'<div class="metric-card"><h3>Knowledge Weight (α)</h3><div class="value">{scheme.alpha}</div></div>',
        unsafe_allow_html=True,
    )
    col_c.markdown(
        f'<div class="metric-card"><h3>Graph Weight (β)</h3><div class="value">{scheme.beta}</div></div>',
        unsafe_allow_html=True,
    )
    col_d.markdown(
        f'<div class="metric-card"><h3>Trend Weight (γ)</h3><div class="value">{scheme.gamma}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "The cards above show the **recommendation strategy** chosen for this user and the "
        "**weights** assigned to each model. "
        "**α (Knowledge)** reflects profile matching, **β (Graph)** reflects what similar users do, "
        "and **γ (Trend)** reflects current popularity. "
        "The system automatically picks a strategy based on how much activity data is available for the user."
    )

    st.markdown("---")

    # ── Event Recommendation Pipeline Visualization ─────
    st.markdown("### 📊 How Event Recommendations Are Built")
    st.markdown(
        "This diagram traces the journey from **raw data** to a **personalised ranked list** of events. "
        "Each layer represents one stage of the pipeline — follow the arrows to see how information "
        "flows from your profile and past behaviour all the way to the final recommendation."
    )

    user_name = user_row["name"]
    fig_pipe, ax_pipe = plt.subplots(figsize=(12, 7))
    ax_pipe.set_xlim(-0.5, 10.5)
    ax_pipe.set_ylim(-0.5, 7.5)
    ax_pipe.axis("off")

    # Box drawing helper
    def _box(ax, x, y, w, h, text, color, fontsize=8, alpha=0.85):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", wrap=True)

    # Arrow helper
    def _arrow(ax, x1, y1, x2, y2, color="#888"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, connectionstyle="arc3,rad=0.05"))

    # ── Layer 1: Data tables (bottom) ────────────────
    tables = [
        (0.3, 0.2, "users.csv\n(profiles)", "#2E86AB"),
        (2.8, 0.2, "events.csv\n(catalog)", "#A23B72"),
        (5.3, 0.2, "attends.csv\n(history)", "#F18F01"),
        (7.8, 0.2, "follows.csv\n(follows)", "#C73E1D"),
    ]
    for tx, ty, label, col in tables:
        _box(ax_pipe, tx, ty, 2.0, 0.9, label, col)

    # ── Layer 2: User profile (middle-left) ──────────
    _box(ax_pipe, 0.5, 2.0, 3.0, 0.9, f"⭐ {user_name}\ninterests • city • culture", "#4A90D9")

    # Arrows from data tables to user profile
    _arrow(ax_pipe, 1.3, 1.1, 1.5, 2.0, "#2E86AB")
    _arrow(ax_pipe, 3.8, 1.1, 2.5, 2.0, "#A23B72")

    # ── Layer 3: Three models (middle) ──────────────
    _box(ax_pipe, 0.0, 3.8, 2.8, 0.9, "Knowledge Model\n(α) interest matching", "#50C878")
    _box(ax_pipe, 3.6, 3.8, 2.8, 0.9, "Graph Model\n(β) similar users", "#9B59B6")
    _box(ax_pipe, 7.2, 3.8, 2.8, 0.9, "Trend Model\n(γ) popularity", "#F39C12")

    # Arrows from user/data to models
    _arrow(ax_pipe, 2.0, 2.9, 1.4, 3.8, "#50C878")  # user -> knowledge
    _arrow(ax_pipe, 2.0, 2.9, 5.0, 3.8, "#9B59B6")  # user -> graph
    _arrow(ax_pipe, 6.3, 1.1, 5.0, 3.8, "#F18F01")  # attends -> graph
    _arrow(ax_pipe, 6.3, 1.1, 8.6, 3.8, "#F39C12")  # attends -> trend
    _arrow(ax_pipe, 8.8, 1.1, 8.6, 3.8, "#C73E1D")  # follows -> trend

    # ── Layer 4: Hybrid ranker ────────────────────
    strat_label = strategy.replace('_', ' ').title()
    _box(ax_pipe, 2.5, 5.5, 5.0, 0.9, f"Hybrid Ranker ({strat_label})\n"
         f"α={scheme.alpha}  β={scheme.beta}  γ={scheme.gamma}", "#E05555")

    # Arrows from models to ranker
    _arrow(ax_pipe, 1.4, 4.7, 4.0, 5.5, "#50C878")
    _arrow(ax_pipe, 5.0, 4.7, 5.0, 5.5, "#9B59B6")
    _arrow(ax_pipe, 8.6, 4.7, 6.0, 5.5, "#F39C12")

    # ── Layer 5: Results ────────────────────────
    _box(ax_pipe, 3.0, 6.8, 4.0, 0.5, f"Top {len(ranked)} Recommended Events", "#2C3E50")
    _arrow(ax_pipe, 5.0, 6.4, 5.0, 6.8, "#E05555")

    ax_pipe.set_title("Event Recommendation Pipeline", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig_pipe)
    plt.close(fig_pipe)

    st.info(
        "**Reading this diagram:** Start at the bottom — the coloured boxes are the data tables "
        "that feed the system. Your *user profile* is extracted from `users.csv`. Three models each "
        "analyse different signals: **Knowledge** matches your interests, **Graph** looks at what "
        "similar users attend, and **Trend** finds what's popular right now. The **Hybrid Ranker** "
        "blends all three scores using the weights shown, producing the final ranked list at the top."
    )

    st.markdown("---")

    # Ranked results table
    st.markdown("### 📋 Ranked Results")
    st.markdown("""
    Each card below is one recommended event, ranked from best to worst.
    Click any card to expand it and see the full details — including
    the city, art forms, ticket price, and a **score breakdown** showing
    exactly how much each model contributed to the final score.
    The coloured tags under "Why recommended" explain why the system
    picked this event for you.
    """)

    for i, (_, r) in enumerate(ranked.iterrows(), 1):
        ev = events.loc[events["event_id"] == r["event_id"]]
        if ev.empty:
            continue
        ev = ev.iloc[0]

        with st.expander(f"**#{i}** — {ev['name']}  |  Score: {r['FinalScore']:.3f}", expanded=(i <= 3)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Event ID:** {r['event_id']}")
                st.markdown(f"**City:** {ev.get('city', '—')}  |  **Venue:** {ev.get('venue', '—')}")
                st.markdown(f"**Art Forms:** {friendly_list(ev.get('art_forms'))}")
                st.markdown(f"**Genres:** {friendly_list(ev.get('genres'))}")
                st.markdown(f"**Type:** {ev.get('event_type', '—')}  |  **Date:** {str(ev.get('date', '—'))[:10]}")
                st.markdown(f"**Ticket Price:** Rs. {ev.get('ticket_price', '—')}")

                explanations = r.get("Explanations", [])
                if isinstance(explanations, list) and explanations:
                    chips = " ".join(f'<span class="explanation-chip">{e}</span>' for e in explanations)
                    st.markdown(f"**Why recommended:** {chips}", unsafe_allow_html=True)

            with col2:
                st.markdown("**Score Breakdown:**")
                scores_data = {
                    "Model": ["Knowledge", "Graph", "Trend", "**Final**"],
                    "Score": [
                        f"{r['KnowledgeScore']:.3f}",
                        f"{r['GraphScore']:.3f}",
                        f"{r['TrendScore']:.3f}",
                        f"**{r['FinalScore']:.3f}**",
                    ],
                }
                st.table(pd.DataFrame(scores_data))

    # Recommendation path graph
    st.markdown("---")
    st.markdown("### 🗺️ Recommendation Path Visualization")
    st.markdown("""
    Each graph traces the logic from **YOU** to the recommended events.
    - **Left:** Content-based paths (interests → categories → events)
    - **Right:** Collaborative paths (similar users → their events)
    """)

    G_cat, cat_recs = build_category_path_graph(selected_user, events, attends)
    G_collab, sim_recs = build_collab_graph(selected_user, events, attends)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Recommendation Paths for {selected_user}",
        fontsize=13, fontweight="bold",
    )
    draw_graph_on_ax(G_cat, selected_user, axes[0],
        "Category-Path Logic\nYOU → events → categories → NEW events")
    draw_graph_on_ax(G_collab, selected_user, axes[1],
        "Collaborative Filtering\nYOU → shared events ← similar users → THEIR events")

    legend_handles = [
        mpatches.Patch(color=USER_COLOR, label="You"),
        mpatches.Patch(color=EVENT_COLOR, label="Attended"),
        mpatches.Patch(color=CATEGORY_COLOR, label="Category"),
        mpatches.Patch(color=SIM_USER_COLOR, label="Similar user"),
        mpatches.Patch(color=RECO_COLOR, label="Recommended"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    st.pyplot(fig)
    plt.close(fig)

    st.info("""
**How to read these graphs :**

**Left graph — "Category-Path Logic":**
The blue circle is **you**. The orange circles are events you have already attended.
From those events the system extracts categories (green circles) like *music*, *dance*, etc.
Then it finds **new events** (red circles) that belong to the same categories.
So the logic is: *"You liked events in these categories → here are more events in those categories."*

**Right graph — "Collaborative Filtering":**
Again the blue circle is **you**. The orange circles are events you attended.
The purple circles are **other users** who attended the same events as you.
The system then looks at what *other* events those similar users attended, and
recommends those (red circles) to you.
So the logic is: *"People who went to the same events as you also went to these → you might like them too."*

**Dashed red arrows** always point toward the recommended events.
""")

    # ── User Interaction Comparison ─────────────────────
    st.markdown("---")
    st.markdown("### 🔀 Compare Event Interactions with Other Users")
    st.markdown("Select 1–4 users to compare **event attendance** and interest overlap.")

    other_user_ids = [uid for uid in users["user_id"] if uid != selected_user]
    compare_options_ev = {
        uid: f"{uid} — {users.loc[users['user_id']==uid, 'name'].iloc[0]} "
             f"({user_counts.get(uid, 0)} events)"
        for uid in other_user_ids
    }
    compare_selected_ev = st.multiselect(
        "Choose users to compare with",
        options=other_user_ids,
        format_func=lambda uid: compare_options_ev[uid],
        max_selections=4,
        key="compare_events_page",
    )

    if compare_selected_ev:
        render_user_comparison(
            selected_user, compare_selected_ev,
            users, events, artists, attends, follows,
            mode="events",
        )
    else:
        st.caption("Pick 1–4 users above to see a side-by-side comparison.")


# =====================================================
# PAGE: Artist Discovery
# =====================================================
elif page == "🎨 Artist Discovery":
    st.markdown("## 🎨 Artist Discovery")
    st.markdown(f"Finding new artists for **{user_row['name']}** based on interests and similar users")

    artist_recs, similar_users_data = run_artist_pipeline(selected_user, _cache_key, top_n=top_n)
    followed_ids = set(follows.loc[follows["user_id"] == selected_user, "artist_id"])

    # ── Artist Recommendation Pipeline Visualization ────
    st.markdown("### 📊 How Artist Recommendations Are Built")
    st.markdown(
        "This diagram traces the journey from **raw data** to a **personalised artist list**. "
        "Follow the arrows to see how your profile, past follows, and similar users combine "
        "into the final recommendation."
    )

    user_name_art = user_row["name"]
    fig_apipe, ax_apipe = plt.subplots(figsize=(12, 7))
    ax_apipe.set_xlim(-0.5, 10.5)
    ax_apipe.set_ylim(-0.5, 7.5)
    ax_apipe.axis("off")

    def _box_a(ax, x, y, w, h, text, color, fontsize=8, alpha=0.85):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", wrap=True)

    def _arrow_a(ax, x1, y1, x2, y2, color="#888"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, connectionstyle="arc3,rad=0.05"))

    # Layer 1: Data tables
    a_tables = [
        (0.3, 0.2, "users.csv\n(profiles)", "#2E86AB"),
        (2.8, 0.2, "artists.csv\n(catalog)", "#A23B72"),
        (5.3, 0.2, "follows.csv\n(follows)", "#F18F01"),
        (7.8, 0.2, "attends.csv\n(history)", "#C73E1D"),
    ]
    for tx, ty, label, col in a_tables:
        _box_a(ax_apipe, tx, ty, 2.0, 0.9, label, col)

    # Layer 2: User profile
    _box_a(ax_apipe, 0.5, 2.0, 3.0, 0.9, f"⭐ {user_name_art}\nart_interests • city", "#4A90D9")

    # Layer 2b: Similar users
    _box_a(ax_apipe, 5.5, 2.0, 3.5, 0.9, "Similar Users\n(Jaccard + Adamic-Adar)", "#9B59B6")

    # Arrows from data to layer 2
    _arrow_a(ax_apipe, 1.3, 1.1, 1.5, 2.0, "#2E86AB")  # users -> profile
    _arrow_a(ax_apipe, 6.3, 1.1, 7.0, 2.0, "#F18F01")  # follows -> similar users
    _arrow_a(ax_apipe, 8.8, 1.1, 7.5, 2.0, "#C73E1D")  # attends -> similar users

    # Layer 3: Two models
    _box_a(ax_apipe, 0.5, 3.8, 3.5, 0.9, "Profile Matching\nart_forms & genres overlap", "#50C878")
    _box_a(ax_apipe, 5.5, 3.8, 3.5, 0.9, "Collaborative Filtering\nsimilar users' follows", "#FF8C42")

    # Arrows from layer 2 to models
    _arrow_a(ax_apipe, 2.0, 2.9, 2.25, 3.8, "#4A90D9")  # profile -> profile matching
    _arrow_a(ax_apipe, 3.8, 1.1, 2.25, 3.8, "#A23B72")  # artists.csv -> profile matching
    _arrow_a(ax_apipe, 7.25, 2.9, 7.25, 3.8, "#9B59B6")  # similar users -> collab

    # Layer 4: Score blender
    _box_a(ax_apipe, 2.5, 5.5, 5.0, 0.9, "Score Blender\n50% Profile + 50% Collab", "#E05555")

    # Arrows from models to blender
    _arrow_a(ax_apipe, 2.25, 4.7, 4.0, 5.5, "#50C878")
    _arrow_a(ax_apipe, 7.25, 4.7, 6.0, 5.5, "#FF8C42")

    # Layer 5: Results
    n_art = len(artist_recs) if artist_recs else 0
    _box_a(ax_apipe, 3.0, 6.8, 4.0, 0.5, f"Top {n_art} Recommended Artists", "#2C3E50")
    _arrow_a(ax_apipe, 5.0, 6.4, 5.0, 6.8, "#E05555")

    ax_apipe.set_title("Artist Recommendation Pipeline", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig_apipe)
    plt.close(fig_apipe)

    st.info(
        "**Reading this diagram:** Start at the bottom — the coloured boxes are the data tables. "
        "Your *user profile* comes from `users.csv`, while *similar users* are found by comparing "
        "attendance and follow patterns (`attends.csv` + `follows.csv`). Two models score each artist: "
        "**Profile Matching** checks if the artist's art forms overlap with your interests, and "
        "**Collaborative Filtering** checks if users similar to you already follow that artist. "
        "The **Score Blender** combines both signals 50/50 to produce the final ranked list at the top."
    )

    st.markdown("---")

    if not artist_recs:
        st.info("No artist recommendations found for this user. Try a user with more activity.")
    else:
        st.markdown("""
        Each card below is a recommended artist ranked by **Final Score**.
        The score combines two things: how well the artist's art forms match
        your interests (**Profile Score**) and how many people similar to you
        already follow this artist (**Collab Score**). Expand a card to see
        the full breakdown and matching interest tags.
        """)

        # Artist cards
        for i, rec in enumerate(artist_recs, 1):
            aid = rec["artist_id"]
            art_row = artists.loc[artists["artist_id"] == aid]
            if art_row.empty:
                continue
            art = art_row.iloc[0]

            with st.expander(
                f"**#{i}** — {art['name']}  |  Score: {rec['FinalArtistScore']:.3f}",
                expanded=(i <= 3),
            ):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**Artist ID:** {aid}")
                    st.markdown(f"**City:** {art.get('city', '—')}")
                    st.markdown(f"**Art Forms:** {friendly_list(art.get('art_forms'))}")
                    st.markdown(f"**Genres:** {friendly_list(art.get('genres'))}")
                    st.markdown(f"**Style:** {friendly_list(art.get('style'))}")

                    shared = rec.get("shared", [])
                    if shared:
                        chips = " ".join(f'<span class="explanation-chip">{s}</span>' for s in shared)
                        st.markdown(f"**Matching interests:** {chips}", unsafe_allow_html=True)

                with col2:
                    st.metric("Profile Score", f"{rec['ProfileScore']:.3f}")
                    st.metric("Collab Score", f"{rec['CollabScore']:.3f}")
                    st.metric("Final Score", f"{rec['FinalArtistScore']:.3f}")

                with col3:
                    st.markdown(f"**Popularity:** {art.get('popularity', '—')}")
                    st.markdown(f"**Followers:** {art.get('follower_count', '—'):,}" if pd.notna(art.get('follower_count')) else "**Followers:** —")
                    st.markdown(f"**Verified:** {'✅' if art.get('verified') else '❌'}")

        # Artist network graph
        st.markdown("---")
        st.markdown("### 🕸️ Artist Discovery Network")
        st.markdown("""
        This graph shows **why** each artist was recommended. It connects you
        to the artists through your interest categories — think of it as a
        visual map of the path: *You → Your Interests → Matching Artists*.
        """)

        G_art = nx.DiGraph()
        G_art.add_node(selected_user, kind="user")
        user_interests = to_tokens(user_row.get("art_interests", ""))

        for cat in list(user_interests)[:5]:
            cn = f"cat:{cat}"
            G_art.add_node(cn, kind="category")
            G_art.add_edge(selected_user, cn, relation="interest")

        for rec in artist_recs[:6]:
            aid = rec["artist_id"]
            G_art.add_node(aid, kind="artist")
            for cat in rec.get("shared", [])[:2]:
                cn = f"cat:{cat}"
                if cn in G_art:
                    G_art.add_edge(cn, aid, relation="recommends")

        fig, ax = plt.subplots(figsize=(12, 7))
        draw_graph_on_ax(G_art, selected_user, ax,
            "Artist Discovery: YOUR interests → matching categories → recommended artists")
        legend_handles = [
            mpatches.Patch(color=USER_COLOR, label="You"),
            mpatches.Patch(color=CATEGORY_COLOR, label="Category"),
            mpatches.Patch(color=ARTIST_COLOR, label="Artist"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9)
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        st.pyplot(fig)
        plt.close(fig)

        st.info("""
**How to read this graph :**

- The **blue circle** in the middle is **you** (the selected user).
- The **green circles** are your **interest categories** — things like music, dance, drama, film, etc. Lines connect you to each category you're interested in.
- The **orange circles** are the **recommended artists**. Each artist is connected to the categories they work in.

**What this tells you:** If you see an artist connected to the same categories you like, that's *why* the system recommended them. Artists with more connections to your interests are a stronger match.

*Example:* If you're interested in "music" and "drama", and an artist performs in both, you'll see two lines connecting that artist to your interests — making it a great match!
""")

        # ── Similar Users Comparison ────────────────────────
        st.markdown("---")
        st.markdown("### 👥 Similar Users Used for Recommendations")
        st.markdown("""
        The system found users who behave most like you — they attend the same
        kinds of events and follow similar artists. The table below shows who
        these users are and how closely they match your activity. Their
        preferences directly influence which artists get recommended to you.
        """)

        if similar_users_data:
            # Summary comparison table
            user_interests_set = to_tokens(user_row.get("art_interests", ""))
            user_attended_count = len(attends.loc[attends["user_id"] == selected_user])
            user_followed_count = len(follows.loc[follows["user_id"] == selected_user])

            comparison_rows = [{
                "": "⭐ YOU",
                "User": selected_user,
                "Name": user_row["name"],
                "City": user_row.get("city", "—"),
                "Art Interests": friendly_list(user_row.get("art_interests")),
                "Events Attended": user_attended_count,
                "Artists Followed": user_followed_count,
                "Similarity": "—",
            }]
            for su in similar_users_data:
                comparison_rows.append({
                    "": "",
                    "User": su["user_id"],
                    "Name": su["name"],
                    "City": su["city"],
                    "Art Interests": friendly_list(su["art_interests"]),
                    "Events Attended": len(su["events_attended"]),
                    "Artists Followed": len(su["artists_followed"]),
                    "Similarity": f"{su['similarity']:.4f}",
                })
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Detailed cards for each similar user
            for idx, su in enumerate(similar_users_data, 1):
                with st.expander(
                    f"**#{idx}** — {su['name']} ({su['user_id']})  |  "
                    f"Similarity: {su['similarity']:.4f}",
                    expanded=(idx <= 2),
                ):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Similarity Scores**")
                        st.metric("Combined Score", f"{su['similarity']:.4f}")
                        st.metric("Jaccard", f"{su['jaccard']:.4f}")
                        st.metric("Adamic-Adar", f"{su['adamic_adar']:.4f}")

                    with c2:
                        st.markdown("**Overlap with You**")
                        if su["shared_interests"]:
                            chips = " ".join(f'<span class="explanation-chip">{s}</span>' for s in su["shared_interests"])
                            st.markdown(f"Shared interests: {chips}", unsafe_allow_html=True)
                        else:
                            st.markdown("_No shared interests_")

                        st.markdown(f"**Shared events:** {len(su['shared_events'])}")
                        if su["shared_events"][:5]:
                            ev_names = []
                            for eid in su["shared_events"][:5]:
                                ev_r = events.loc[events["event_id"] == eid]
                                ev_names.append(ev_r.iloc[0]["name"] if not ev_r.empty else str(eid))
                            st.markdown(", ".join(ev_names))

                        st.markdown(f"**Shared followed artists:** {len(su['shared_artists'])}")
                        if su["shared_artists"][:5]:
                            art_names = []
                            for aid in su["shared_artists"][:5]:
                                a_r = artists.loc[artists["artist_id"] == aid]
                                art_names.append(a_r.iloc[0]["name"] if not a_r.empty else str(aid))
                            st.markdown(", ".join(art_names))

                    with c3:
                        st.markdown("**Their Activity**")
                        st.markdown(f"**City:** {su['city']}")
                        st.markdown(f"**Events attended:** {len(su['events_attended'])}")
                        st.markdown(f"**Artists followed:** {len(su['artists_followed'])}")

                        # Show unique artists this user follows that you don't
                        unique_artists = set(su["artists_followed"]) - followed_ids
                        st.markdown(f"**Artists you haven't discovered:** {len(unique_artists)}")
                        if unique_artists:
                            sample = list(unique_artists)[:4]
                            names = []
                            for aid in sample:
                                a_r = artists.loc[artists["artist_id"] == aid]
                                names.append(a_r.iloc[0]["name"] if not a_r.empty else str(aid))
                            st.caption(", ".join(names) + ("…" if len(unique_artists) > 4 else ""))
        else:
            st.warning("No similar users found — this user may be a cold-start user with very little activity.")

        # ── Recommendation Sources ──────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 How Each Artist Recommendation Was Built")
        st.markdown("""
        Each recommended artist was scored using **two signals**. This table shows which
        similar users' data contributed to the **collaborative score** for each artist.
        """)

        source_rows = []
        for rec in artist_recs:
            aid = rec["artist_id"]
            art_r = artists.loc[artists["artist_id"] == aid]
            art_name = art_r.iloc[0]["name"] if not art_r.empty else aid

            profile_reason = (
                f"Matches your interests: {', '.join(rec['shared'])}"
                if rec["shared"] else "—"
            )

            sources = rec.get("sources", [])
            if sources:
                collab_reason = "; ".join(
                    f"{s['name']} (sim={s['similarity']:.3f})"
                    for s in sorted(sources, key=lambda x: x["similarity"], reverse=True)[:3]
                )
            else:
                collab_reason = "—"

            source_rows.append({
                "Artist": art_name,
                "Profile Score": f"{rec['ProfileScore']:.3f}",
                "Why (Profile)": profile_reason,
                "Collab Score": f"{rec['CollabScore']:.3f}",
                "Contributed By (Similar Users)": collab_reason,
                "Final Score": f"{rec['FinalArtistScore']:.3f}",
            })

        source_df = pd.DataFrame(source_rows)
        st.dataframe(source_df, use_container_width=True, hide_index=True)

        st.info("""
**How to read this table:**

- **Profile Score** comes from matching your `art_interests` against each artist's `art_forms` and `genres` — this is purely your taste profile.
- **Collab Score** comes from **similar users**: if users with similar behaviour follow an artist, that artist gets a boost. The "Contributed By" column shows which similar users drove that score.
- **Final Score** = 50% Profile + 50% Collab. Artists that score well on *both* signals rank highest.
""")

    # ── User Interaction Comparison (Artist Discovery) ──
    st.markdown("---")
    st.markdown("### 🔀 Compare Artist Interactions with Other Users")
    st.markdown("Select 1–4 users to compare **artist follows** and interest overlap.")

    other_user_ids_art = [uid for uid in users["user_id"] if uid != selected_user]
    compare_options_art = {
        uid: f"{uid} — {users.loc[users['user_id']==uid, 'name'].iloc[0]} "
             f"({user_counts.get(uid, 0)} events)"
        for uid in other_user_ids_art
    }
    compare_selected_art = st.multiselect(
        "Choose users to compare with",
        options=other_user_ids_art,
        format_func=lambda uid: compare_options_art[uid],
        max_selections=4,
        key="compare_artist_page",
    )

    if compare_selected_art:
        render_user_comparison(
            selected_user, compare_selected_art,
            users, events, artists, attends, follows,
            mode="artists",
        )
    else:
        st.caption("Pick 1–4 users above to see a side-by-side comparison.")


# =====================================================
# PAGE: Model Comparison
# =====================================================
elif page == "📊 Model Comparison":
    st.markdown("## 📊 Model-by-Model Comparison")
    st.markdown("""
    Compare how each of the three models scores events independently,
    and how the hybrid ranker combines them into a final score.
    """)

    ranked, k_scores, g_scores, t_scores, strategy, n_interactions = run_recommendation_pipeline(
        selected_user, _cache_key, top_n=top_n,
    )

    # Score distribution chart
    st.markdown("### 📉 Score Distribution Across Models")

    chart_data = ranked[["event_id", "KnowledgeScore", "GraphScore", "TrendScore", "FinalScore"]].copy()
    chart_data = chart_data.merge(events[["event_id", "name"]], on="event_id", how="left")
    chart_data["label"] = chart_data["name"].str[:25]
    chart_data = chart_data.set_index("label")

    st.bar_chart(
        chart_data[["KnowledgeScore", "GraphScore", "TrendScore"]],
        use_container_width=True,
    )

    st.info(f"""
**Reading this chart:** Each group of bars represents one recommended event. The three colours correspond to scores from each model:

- **KnowledgeScore** — how well the event matches the user's profile (art interests, city, culture preferences).
- **GraphScore** — how strongly the event is connected to similar users in the interaction graph (collaborative signal).
- **TrendScore** — how popular / fast-growing the event is across all users in a recent time window.

Taller bars mean higher relevance under that model. The **Hybrid Ranker** combines these three scores (using the weight strategy shown below) into the single **FinalScore** used for the final ranking.
""")

    # Detailed comparison table
    st.markdown("### 📋 Detailed Event Score Table")
    st.markdown("""
    This table lists every recommended event with its individual scores from each model
    and the combined final score. The "Final" column is highlighted — darker shades
    mean higher scores (i.e. stronger recommendations). You can sort any column by clicking its header.
    """)
    display_df = ranked[["event_id", "KnowledgeScore", "GraphScore", "TrendScore", "FinalScore"]].copy()
    display_df = display_df.merge(events[["event_id", "name", "city", "art_forms"]], on="event_id", how="left")
    display_df["art_forms"] = display_df["art_forms"].apply(friendly_list)
    display_df = display_df[["event_id", "name", "city", "art_forms", "KnowledgeScore", "GraphScore", "TrendScore", "FinalScore"]]
    display_df.columns = ["ID", "Event Name", "City", "Art Forms", "Knowledge", "Graph", "Trend", "Final"]
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"

    st.dataframe(
        display_df.style.format({
            "Knowledge": "{:.3f}", "Graph": "{:.3f}", "Trend": "{:.3f}", "Final": "{:.3f}",
        }).background_gradient(subset=["Final"], cmap="YlOrRd"),
        use_container_width=True,
    )

    # Artist score table
    st.markdown("### 🎨 Detailed Artist Score Table")
    st.markdown("""
    Artist recommendations use two models: **Profile Matching** (interest overlap) and
    **Collaborative Filtering** (similar users' follows). The final score blends both at 50/50.
    """)

    artist_recs_mc, _ = run_artist_pipeline(selected_user, _cache_key, top_n=top_n)

    if artist_recs_mc:
        art_rows = []
        for rec in artist_recs_mc:
            aid = rec["artist_id"]
            art_r = artists.loc[artists["artist_id"] == aid]
            if art_r.empty:
                continue
            art = art_r.iloc[0]
            art_rows.append({
                "ID": aid,
                "Artist Name": art.get("name", "?"),
                "City": art.get("city", "—"),
                "Art Forms": friendly_list(art.get("art_forms")),
                "Profile": rec["ProfileScore"],
                "Collab": rec["CollabScore"],
                "Final": rec["FinalArtistScore"],
            })
        art_display_df = pd.DataFrame(art_rows)
        art_display_df.index = range(1, len(art_display_df) + 1)
        art_display_df.index.name = "Rank"

        st.dataframe(
            art_display_df.style.format({
                "Profile": "{:.3f}", "Collab": "{:.3f}", "Final": "{:.3f}",
            }).background_gradient(subset=["Final"], cmap="YlOrRd"),
            use_container_width=True,
        )

        # Artist score chart
        st.markdown("### 📉 Artist Score Distribution")
        art_chart = art_display_df[["Artist Name", "Profile", "Collab"]].copy()
        art_chart["label"] = art_chart["Artist Name"].str[:25]
        art_chart = art_chart.set_index("label")[["Profile", "Collab"]]
        st.bar_chart(art_chart, use_container_width=True)

        st.info("""
**Reading this chart:** Each group of bars represents one recommended artist:

- **Profile** — how well the artist's `art_forms` and `genres` match your personal `art_interests` (content-based signal).
- **Collab** — how strongly similar users follow this artist (collaborative signal from the interaction graph).

The **Final Artist Score** = 50% Profile + 50% Collab.
""")
    else:
        st.info("No artist recommendations available for this user.")

    # Weights visualization
    st.markdown("### ⚖️ Hybrid Weight Configuration")
    st.markdown("""
    The system doesn't treat every model equally — it adjusts how much it
    trusts each model depending on how much data it has about a user.

    - **New users** (cold-start): The system relies more on matching interests
      because it doesn't know much about the user's behaviour yet.
    - **Active users**: The system shifts trust toward what similar users do
      (the Graph model) since there's enough activity history to find patterns.

    The three Greek letters (α, β, γ) are the "importance weights" for
    Knowledge, Graph, and Trend respectively. They always add up to 1.0 (100%).
    """)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cold-Start Strategy** (< 5 interactions)")
        cs = DEFAULT_WEIGHTS["cold_start"]
        st.markdown(f"- Knowledge (α): **{cs.alpha}**")
        st.markdown(f"- Graph (β): **{cs.beta}**")
        st.markdown(f"- Trend (γ): **{cs.gamma}**")

    with col2:
        st.markdown("**Active Strategy** (≥ 5 interactions)")
        ac = DEFAULT_WEIGHTS["active"]
        st.markdown(f"- Knowledge (α): **{ac.alpha}**")
        st.markdown(f"- Graph (β): **{ac.beta}**")
        st.markdown(f"- Trend (γ): **{ac.gamma}**")

    scheme = DEFAULT_WEIGHTS[strategy]
    st.info(
        f"**Current user strategy: {strategy.replace('_',' ').title()}** "
        f"({n_interactions} interactions) → "
        f"α={scheme.alpha}, β={scheme.beta}, γ={scheme.gamma}"
    )

    # Weight pie charts
    fig_w, axes_w = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (strat_name, ws) in zip(axes_w, [("Cold Start", DEFAULT_WEIGHTS["cold_start"]), ("Active", DEFAULT_WEIGHTS["active"])]):
        vals = [ws.alpha, ws.beta, ws.gamma]
        labels = [f"Knowledge\n(α={ws.alpha})", f"Graph\n(β={ws.beta})", f"Trend\n(γ={ws.gamma})"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        ax.pie(vals, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
        ax.set_title(f"{strat_name} Strategy", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_w)
    plt.close(fig_w)

    st.info("""
**How to read the pie charts:**

Each pie chart shows the share of influence each model gets in the final recommendation.
The bigger the slice, the more that model's opinion counts.

- **Red (Knowledge)** = matching user interests to event/artist attributes.
- **Teal (Graph)** = what similar users have attended or followed.
- **Blue (Trend)** = what's popular and growing right now.

For a *new user* with little history, the red slice (interests) is biggest because
that's all we know about them. For an *active user*, the teal slice (similar users)
grows because we have enough behaviour data to find reliable patterns.
""")


# =====================================================
# PAGE: Evaluation Metrics
# =====================================================
elif page == "🔬 Evaluation Metrics":
    st.markdown("## 🔬 Offline Evaluation Metrics")
    st.markdown("""
    This page answers the question: **"How good are the recommendations?"**

    We test the system by hiding each user's most recent event attendance and then
    asking the model to recommend events. If the hidden event shows up in the
    recommendations, the model got it right. This "hide-and-check" method is called
    **leave-last-out** evaluation — it simulates a real scenario where the system
    has to predict what a user will do next.
    """)

    with st.spinner("Running evaluation across users... (this may take a moment)"):
        attends_sorted = attends.sort_values("timestamp")
        test = attends_sorted.groupby("user_id").tail(1)
        train = attends_sorted.drop(test.index)

        km = KnowledgeMatcher(budget_col=None)
        km.fit(users, events)
        trend_model = TrendWindowRecommender().fit(train)
        trend_df = trend_model.recommend(top_n=top_n * 3, window_days=14)
        t_scores_eval = trend_df[["event_id", "TrendScore"]]

        ranker = HybridRanker()
        rec_map: dict[str, list[str]] = {}
        rel_map: dict[str, set[str]] = {}

        test_users = test["user_id"].unique()
        sample_users = test_users[:min(30, len(test_users))]

        progress = st.progress(0)
        for idx, uid in enumerate(sample_users):
            rel_map[uid] = set(test.loc[test["user_id"] == uid, "event_id"].astype(str))

            try:
                km_df = km.recommend(uid, top_n=top_n * 3)
                k_sc = km_df[["event_id", "KnowledgeScore"]]
            except Exception:
                k_sc = pd.DataFrame(columns=["event_id", "KnowledgeScore"])

            try:
                g_df = recommend_from_similar_users(train, follows, uid, top_users=20, top_n=top_n * 2, alpha=0.5)
                g_sc = g_df[["event_id", "GraphScore"]] if not g_df.empty else pd.DataFrame(columns=["event_id", "GraphScore"])
            except Exception:
                g_sc = pd.DataFrame(columns=["event_id", "GraphScore"])

            cand = pd.DataFrame({"event_id": pd.unique(
                pd.concat([k_sc["event_id"], g_sc.get("event_id", pd.Series(dtype=str)), t_scores_eval.get("event_id", pd.Series(dtype=str))], ignore_index=True)
            )})
            m = cand.merge(k_sc, on="event_id", how="left")
            m = m.merge(g_sc, on="event_id", how="left")
            m = m.merge(t_scores_eval, on="event_id", how="left")
            m[["KnowledgeScore", "GraphScore", "TrendScore"]] = m[["KnowledgeScore", "GraphScore", "TrendScore"]].fillna(0.0)

            ui = len(train.loc[train["user_id"] == uid])
            r = ranker.rank(m, user_interactions=ui, top_n=top_n)
            rec_map[uid] = r["event_id"].astype(str).tolist()
            progress.progress((idx + 1) / len(sample_users))

        progress.empty()

        catalog = set(events["event_id"].astype(str))
        avg_p = np.mean([precision_at_k(rec_map[u], rel_map[u], top_n) for u in sample_users])
        avg_r = np.mean([recall_at_k(rec_map[u], rel_map[u], top_n) for u in sample_users])
        avg_n = np.mean([ndcg_at_k(rec_map[u], rel_map[u], top_n) for u in sample_users])
        cov = coverage(rec_map, catalog)

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(
        f'<div class="metric-card"><h3>Precision@{top_n}</h3><div class="value">{avg_p:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-card"><h3>Recall@{top_n}</h3><div class="value">{avg_r:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-card"><h3>NDCG@{top_n}</h3><div class="value">{avg_n:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    col4.markdown(
        f'<div class="metric-card"><h3>Catalog Coverage</h3><div class="value">{cov:.2%}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Metric explanations
    st.markdown("### 📖 What Do These Metrics Mean?")
    st.markdown(f"""
    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | **Precision@{top_n}** | {avg_p:.4f} | Of the {top_n} items recommended, {avg_p:.1%} were actually relevant |
    | **Recall@{top_n}** | {avg_r:.4f} | Of all relevant items, {avg_r:.1%} were found in the top {top_n} |
    | **NDCG@{top_n}** | {avg_n:.4f} | Quality of ranking — higher means relevant items are ranked higher |
    | **Coverage** | {cov:.2%} | {int(cov * len(catalog))}/{len(catalog)} unique events were recommended across all users |
    """)

    st.markdown(f"""
    > **Evaluation method:** Leave-last-out on **{len(sample_users)} users**.
    > Each user's chronologically last event attendance is held out as ground truth.
    > The model recommends Top-{top_n} events using the remaining history,
    > and we check if the held-out event appears in the recommendations.
    """)

    # Per-user breakdown
    st.markdown("### 👥 Per-User Precision Breakdown")
    per_user = []
    for uid in sample_users:
        p = precision_at_k(rec_map[uid], rel_map[uid], top_n)
        name = users.loc[users["user_id"] == uid, "name"]
        name = name.iloc[0] if not name.empty else "?"
        n_train = len(train.loc[train["user_id"] == uid])
        per_user.append({"User": uid, "Name": name, "Train Events": n_train, f"Precision@{top_n}": p})

    per_user_df = pd.DataFrame(per_user)

    st.info(f"""
**What does this table show?**

Each row is one user from the evaluation sample. The columns tell you:

- **Train Events** — how many past attendances the model had to learn from. Fewer events = harder to predict ("cold-start").
- **Precision@{top_n}** — did the user's *held-out* event appear in the top-{top_n} recommendations?
  A value of **1.0** means *yes, the model got it right*; **0.0** means *no, it missed*.

Green-shaded rows are correct predictions; red-shaded rows are misses. A higher overall average means the model is doing well at predicting what users will attend next.
""")

    st.dataframe(
        per_user_df.style.format({f"Precision@{top_n}": "{:.4f}"})
        .background_gradient(subset=[f"Precision@{top_n}"], cmap="RdYlGn"),
        use_container_width=True,
    )


# =====================================================
# PAGE: Data Explorer
# =====================================================
elif page == "📈 Data Explorer":
    st.markdown("## 📈 Data Explorer")
    st.markdown(
        "Browse the **raw data** that powers the recommendation system. "
        "Each tab below covers a different part of the dataset — users, events, artists, attendance records, and follow relationships. "
        "Use this section to understand the data landscape: how many records exist, how they are distributed, and what the underlying information looks like."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Users", "🎪 Events", "🎨 Artists", "📅 Attends", "❤️ Follows"])

    with tab1:
        st.markdown(f"**{len(users)} users** in the dataset")
        st.markdown(
            "This tab shows the **user base** of the platform. "
            "The bar chart below displays the **top 15 cities** where users are located — "
            "taller bars mean more users live in that city. "
            "The table underneath lists individual user profiles including their art interests, city, and cultural preferences."
        )
        # City distribution
        city_counts = users["city"].value_counts().head(15)
        st.bar_chart(city_counts)
        st.dataframe(users.head(30), use_container_width=True)

    with tab2:
        st.markdown(f"**{len(events)} events** in the dataset")
        st.markdown(
            "This tab explores the **cultural events** available in the system. "
            "The bar chart shows how events are split across different **event types** (e.g. festivals, workshops, exhibitions) — "
            "taller bars mean that type of event is more common. "
            "The table below lists event details including location, art forms, genres, dates, and ticket prices."
        )
        # Event type distribution
        type_counts = events["event_type"].value_counts()
        st.bar_chart(type_counts)
        st.dataframe(events[["event_id", "name", "city", "art_forms", "genres", "event_type", "date", "ticket_price", "status"]].head(30), use_container_width=True)

    with tab3:
        st.markdown(f"**{len(artists)} artists** in the dataset")
        st.markdown(
            "This tab shows the **artists** registered on the platform. "
            "The bar chart breaks down artists by their **popularity level** (e.g. low, medium, high) — "
            "taller bars indicate more artists fall into that popularity bracket. "
            "The table below provides artist profiles with their art forms, genres, city, follower count, and verification status."
        )
        pop_counts = artists["popularity"].value_counts()
        st.bar_chart(pop_counts)
        st.dataframe(artists[["artist_id", "name", "art_forms", "genres", "city", "popularity", "follower_count", "verified"]].head(30), use_container_width=True)

    with tab4:
        st.markdown(f"**{len(attends)} attendance records**")
        st.markdown(
            "This tab tracks **event attendance** over time. "
            "The line chart shows the **number of people who attended events each month** — "
            "rising lines mean attendance grew during that period, while dips indicate quieter months. "
            "This helps identify seasonal trends and the overall growth of event participation. "
            "The table below lists individual attendance records with timestamps and RSVP statuses."
        )
        # Attendance over time
        attends_monthly = attends.set_index("timestamp").resample("M").size()
        st.line_chart(attends_monthly)
        st.dataframe(attends.head(30), use_container_width=True)

    with tab5:
        st.markdown(f"**{len(follows)} follow relationships**")
        st.markdown(
            "This tab shows which **artists are most popular** based on user follows. "
            "The bar chart displays the **top 15 most-followed artists** — "
            "taller bars mean more users have chosen to follow that artist, indicating higher fan engagement. "
            "The table below lists individual follow records showing which users follow which artists."
        )
        top_followed = follows["artist_id"].value_counts().head(15)
        top_followed.index = [
            artists.loc[artists["artist_id"] == aid, "name"].iloc[0]
            if not artists.loc[artists["artist_id"] == aid].empty
            else aid
            for aid in top_followed.index
        ]
        st.bar_chart(top_followed)
        st.dataframe(follows.head(30), use_container_width=True)


# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#888; font-size:0.85rem;">'
    '🎭 Cultural Event Recommendation System — Sri Lanka &nbsp;|&nbsp; '
    'Hybrid Model: Knowledge + Graph + Trend &nbsp;|&nbsp; '
    'Built with ❤️ for the Arts Community<br>'
    '</div>',
    unsafe_allow_html=True,
)
