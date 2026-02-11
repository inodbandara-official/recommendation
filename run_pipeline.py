"""
=============================================================
  Recommendation Pipeline — Interactive Step Navigator
=============================================================

Run:  python run_pipeline.py

Select a step number (1-7) to execute that step, or run all
steps sequentially. No need to type full file paths.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = {
    "1": {
        "file": "pipeline/step1_load_data.py",
        "title": "Dataset Overview",
        "desc": "Loads all 5 CSVs, prints node/edge counts, sample rows, discovered categories.",
    },
    "2": {
        "file": "pipeline/step2_build_graph.py",
        "title": "Build & Visualise Graph",
        "desc": "Builds heterogeneous graph (2,527 nodes, 10,417 edges). Saves pipeline/step2_graph.png.",
    },
    "3": {
        "file": "pipeline/step3_basic_reco.py",
        "title": "Basic Recommendations",
        "desc": "Category-path and similar-user recommendations with dual-panel graph.",
    },
    "4": {
        "file": "pipeline/step4_run_model.py",
        "title": "Full Model Pipeline + Evaluation",
        "desc": "Runs knowledge + graph + trend models, hybrid blend, explanations, offline metrics.",
    },
    "5": {
        "file": "pipeline/step5_highlight_paths.py",
        "title": "Highlight Recommendation Paths",
        "desc": "Traces and visualises colour-coded paths from user to each recommendation.",
    },
    "6": {
        "file": "pipeline/step6_dynamic_input.py",
        "title": "Interactive Dynamic Input",
        "desc": "Pick users, filter categories, compare users side-by-side. Interactive mode.",
    },
    "7": {
        "file": "pipeline/step7_advanced_graph.py",
        "title": "Advanced Graph + Scoreboard",
        "desc": "Multi-path graph, combined scores, degree centrality, scoreboard.",
    },
}


def print_banner() -> None:
    print()
    print("=" * 62)
    print("   RECOMMENDATION SYSTEM — Step Navigator")
    print("=" * 62)
    print()


def print_menu() -> None:
    for num, info in STEPS.items():
        print(f"    {num}. {info['title']}")
        print(f"       {info['desc']}")
        print()
    print("    8. Run ALL steps (1-7) sequentially")
    print("    0. Exit")
    print()


def run_step(step_num: str) -> None:
    info = STEPS[step_num]
    script = ROOT / info["file"]

    if not script.exists():
        print(f"\n    ERROR: {info['file']} not found.\n")
        return

    print()
    print("-" * 62)
    print(f"  Running Step {step_num}: {info['title']}")
    print(f"  Script: {info['file']}")
    print("-" * 62)
    print()

    env = os.environ.copy()
    # Ensure src package is importable (needed for step 4)
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
        choice = input("    Select step [1-7], 8 for all, 0 to exit: ").strip()

        if choice == "0":
            print("\n    Goodbye!\n")
            break
        elif choice == "8":
            for step_num in STEPS:
                run_step(step_num)
            print("\n    All 7 steps completed.\n")
        elif choice in STEPS:
            run_step(choice)
            print()
        else:
            print("\n    Invalid choice. Enter 1-7, 8, or 0.\n")


if __name__ == "__main__":
    main()
