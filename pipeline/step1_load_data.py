"""
=============================================================
  RECOMMENDATION SYSTEM: Graph-Based Recommendation Engine
  Step 1 — Load and Understand the Real Dataset
=============================================================

Run:  python pipeline/step1_load_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def load_all() -> dict[str, pd.DataFrame]:
    files = {
        "users": "users.csv",
        "events": "events.csv",
        "artists": "artists.csv",
        "attends": "attends.csv",
        "follows": "follows.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, filename in files.items():
        frames[name] = pd.read_csv(DATA_DIR / filename)
    return frames


def to_tokens(val: object) -> set[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return set()
    s = str(val).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return {p.strip().strip("'\" ").lower() for p in s.split(",") if p.strip().strip("'\" ")}


def extract_categories(events: pd.DataFrame) -> set[str]:
    cats: set[str] = set()
    for col in ("art_forms", "genres"):
        if col in events.columns:
            for val in events[col].dropna():
                cats.update(to_tokens(val))
    return cats


def print_banner(text: str) -> None:
    width = 60
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_section(title: str) -> None:
    print(f"\n  --- {title} ---\n")


def main() -> None:
    data = load_all()

    users = data["users"]
    events = data["events"]
    artists = data["artists"]
    attends = data["attends"]
    follows = data["follows"]

    # ── Banner ──────────────────────────────────────────────
    print_banner("STEP 1: Dataset Overview")

    # ── Node summary ────────────────────────────────────────
    categories = extract_categories(events)

    print_section("Node Counts")
    print(f"    User nodes      :  {len(users):,}")
    print(f"    Event nodes     :  {len(events):,}")
    print(f"    Artist nodes    :  {len(artists):,}")
    print(f"    Category tokens :  {len(categories):,}")

    # ── Edge / interaction summary ──────────────────────────
    print_section("Interaction Edges")
    print(f"    Attends (user ➜ event)    :  {len(attends):,}")
    print(f"    Follows (user ➜ artist)   :  {len(follows):,}")

    # event→artist links from artist_ids column
    event_artist_count = 0
    if "artist_ids" in events.columns:
        for val in events["artist_ids"].dropna():
            event_artist_count += len(to_tokens(val))
    print(f"    Performs (event ➜ artist)  :  {event_artist_count:,}")
    print(f"    Total edges               :  {len(attends) + len(follows) + event_artist_count:,}")

    # ── User snapshot ───────────────────────────────────────
    print_section("Sample Users (first 5)")
    print(users[["user_id", "name", "art_interests", "region_preference"]].head(5).to_string(index=False))

    # ── Event snapshot ──────────────────────────────────────
    print_section("Sample Events (first 5)")
    print(events[["event_id", "name", "art_forms", "genres", "region"]].head(5).to_string(index=False))

    # ── Artist snapshot ─────────────────────────────────────
    print_section("Sample Artists (first 5)")
    print(artists[["artist_id", "name", "art_forms", "genres", "popularity"]].head(5).to_string(index=False))

    # ── Attend snapshot ─────────────────────────────────────
    print_section("Sample Interactions — Attends (first 5)")
    print(attends.head(5).to_string(index=False))

    # ── Follow snapshot ─────────────────────────────────────
    print_section("Sample Interactions — Follows (first 5)")
    print(follows.head(5).to_string(index=False))

    # ── Categories discovered ───────────────────────────────
    print_section("Discovered Categories / Genres")
    sorted_cats = sorted(categories)
    for i in range(0, len(sorted_cats), 5):
        chunk = sorted_cats[i : i + 5]
        print("    " + ",  ".join(chunk))

    print_banner("End of Step 1")
    print()


if __name__ == "__main__":
    main()
