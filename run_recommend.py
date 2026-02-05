"""Run hybrid recommendations for a user."""
from pathlib import Path

import pandas as pd

from src.hybrid import recommend_events, attach_explanations
from src.trend_based import TrendWindowRecommender
from src.graph_based import recommend_from_similar_users


DATA_DIR = Path("data")


def run_hybrid() -> None:
    """Full hybrid recommendations with explanations."""
    user_id = input("Enter user_id (e.g. U0008): ").strip() or "U0008"
    top_n = int(input("Enter top_n (default 10): ").strip() or "10")

    print(f"\nGenerating top {top_n} hybrid recommendations for {user_id}...\n")
    recs = recommend_events(user_id=user_id, top_n=top_n, data_dir=DATA_DIR)
    print(recs[["event_id", "KnowledgeScore", "GraphScore", "TrendScore", "FinalScore", "Explanations"]].to_string())


def run_trend_only() -> None:
    """Trend-based recommendations (no user required)."""
    top_n = int(input("Enter top_n (default 10): ").strip() or "10")
    window_days = int(input("Enter window_days (default 14): ").strip() or "14")

    print(f"\nGenerating top {top_n} trending events (last {window_days} days)...\n")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    trend = TrendWindowRecommender().fit(attends)
    result = trend.recommend(top_n=top_n, window_days=window_days)
    print(result.to_string())


def run_graph_only() -> None:
    """Graph-based recommendations using similar users."""
    user_id = input("Enter user_id (e.g. U0008): ").strip() or "U0008"
    top_n = int(input("Enter top_n (default 10): ").strip() or "10")

    print(f"\nGenerating top {top_n} graph-based recommendations for {user_id}...\n")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    follows = pd.read_csv(DATA_DIR / "follows.csv")
    result = recommend_from_similar_users(attends, follows, target_user=user_id, top_n=top_n)
    if result.empty:
        print("No recommendations found (user may have no interactions).")
    else:
        print(result.to_string())


def run_with_explanations() -> None:
    """Add explanations to hybrid recommendations."""
    user_id = input("Enter user_id (e.g. U0008): ").strip() or "U0008"
    top_n = int(input("Enter top_n (default 10): ").strip() or "10")
    interests_raw = input("Enter interests (comma-separated, e.g. music,dance): ").strip()
    user_interests = [i.strip() for i in interests_raw.split(",") if i.strip()] or None
    user_region = input("Enter region (e.g. north_western, leave blank to skip): ").strip() or None

    print(f"\nGenerating recommendations with custom explanations for {user_id}...\n")
    recs = recommend_events(user_id=user_id, top_n=top_n, data_dir=DATA_DIR)
    events = pd.read_csv(DATA_DIR / "events.csv")
    out = attach_explanations(recs, events=events, user_interests=user_interests, user_region=user_region)
    print(out[["event_id", "FinalScore", "Explanations"]].to_string())


def main() -> None:
    print("\n=== Hybrid Recommendation System ===")
    print("1. Hybrid recommendations (knowledge + graph + trend)")
    print("2. Trend-only recommendations")
    print("3. Graph-only recommendations (similar users)")
    print("4. Hybrid with custom explanations")
    print("0. Exit")

    choice = input("\nSelect an option [1-4, 0 to exit]: ").strip()

    if choice == "1":
        run_hybrid()
    elif choice == "2":
        run_trend_only()
    elif choice == "3":
        run_graph_only()
    elif choice == "4":
        run_with_explanations()
    elif choice == "0":
        print("Goodbye!")
    else:
        print("Invalid option. Please enter 1-4 or 0.")


if __name__ == "__main__":
    main()
