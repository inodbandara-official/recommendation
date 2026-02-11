"""
=============================================================
  Recommendation Pipeline — Interactive Navigator
=============================================================

Run:  python run_pipeline.py

Select a number (1-7) to execute that section, or run all
sections sequentially. No need to type full file paths.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SECTIONS = {
    "1": {
        "file": "pipeline/1_load_data.py",
        "title": "Dataset Overview",
        "desc": "Loads all 5 CSVs, prints node/edge counts, sample rows, discovered categories.",
    },
    "2": {
        "file": "pipeline/2_build_graph.py",
        "title": "Build & Visualise Graph",
        "desc": "Builds heterogeneous graph (2,527 nodes, 10,417 edges). Saves pipeline/2_graph.png.",
    },
    "3": {
        "file": "pipeline/3_basic_reco.py",
        "title": "Basic Recommendations",
        "desc": "Category-path and similar-user recommendations with dual-panel graph.",
    },
    "4": {
        "file": "pipeline/4_run_model.py",
        "title": "Full Model Pipeline + Evaluation",
        "desc": "Runs knowledge + graph + trend models, hybrid blend, explanations, offline metrics.",
    },
    "5": {
        "file": "pipeline/5_highlight_paths.py",
        "title": "Highlight Recommendation Paths",
        "desc": "Traces and visualises colour-coded paths from user to each recommendation.",
    },
    "6": {
        "file": "pipeline/6_dynamic_input.py",
        "title": "Interactive Dynamic Input",
        "desc": "Pick users, filter categories, compare users side-by-side. Interactive mode.",
    },
    "7": {
        "file": "pipeline/7_advanced_graph.py",
        "title": "Advanced Graph + Scoreboard",
        "desc": "Multi-path graph, combined scores, degree centrality, scoreboard.",
    },
}


def print_banner() -> None:
    print()
    print("=" * 62)
    print("   RECOMMENDATION SYSTEM — Pipeline Navigator")
    print("=" * 62)
    print()


def print_menu() -> None:
    for num, info in SECTIONS.items():
        print(f"    {num}. {info['title']}")
        print(f"       {info['desc']}")
        print()
    print("    8. Run ALL (1-7) sequentially")
    print("    0. Exit")
    print()


def run_section(num: str) -> None:
    info = SECTIONS[num]
    script = ROOT / info["file"]

    if not script.exists():
        print(f"\n    ERROR: {info['file']} not found.\n")
        return

    print()
    print("-" * 62)
    print(f"  Running {num}: {info['title']}")
    print(f"  Script: {info['file']}")
    print("-" * 62)
    print()

    env = os.environ.copy()
    # Ensure src package is importable
    python_path = env.get("PYTHONPATH", "")
    if str(ROOT) not in python_path:
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{python_path}" if python_path else str(ROOT)

    try:
        subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            env=env,
        )
    except KeyboardInterrupt:
        print("\n    (interrupted)")


def main() -> None:
    print_banner()

    while True:
        print_menu()
        choice = input("    Select [1-7], 8 for all, 0 to exit: ").strip()

        if choice == "0":
            print("\n    Goodbye!\n")
            break
        elif choice == "8":
            for num in SECTIONS:
                run_section(num)
            print("\n    All 7 sections completed.\n")
        elif choice in SECTIONS:
            run_section(choice)
            print()
        else:
            print("\n    Invalid choice. Enter 1-7, 8, or 0.\n")


if __name__ == "__main__":
    main()
