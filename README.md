# Hybrid Recommendation System

A hybrid cultural event recommendation system for Sri Lanka, combining knowledge-based, graph-based, and trend-based approaches.

---

## Getting Started (From Scratch)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/Recommendation.git
cd Recommendation
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3 — Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\activate
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -e .[dev]
```

### Step 5 — Verify Data Files

Ensure these 5 CSV files exist in the `data/` folder:

| File | Description |
|------|-------------|
| `users.csv` | 1,501 users with art_interests, region_preference |
| `events.csv` | 1,001 events with art_forms, genres, region, ticket_price |
| `artists.csv` | 501 artists with art_forms, genres, popularity |
| `attends.csv` | 7,197 user-event attendance records |
| `follows.csv` | 12,961 user-artist follow records |

### Step 6 — Run the System

You have **two entry points**:

#### Option A: Recommendation Engine (Interactive Menu)

```bash
python run_recommend.py
```

```
=== Hybrid Recommendation System ===
1. Hybrid recommendations (knowledge + graph + trend)
2. Trend-only recommendations
3. Graph-only recommendations (similar users)
4. Hybrid with custom explanations
0. Exit
```

#### Option B: System Pipeline (7-Step Walkthrough)

```bash
python run_pipeline.py
```

```
==============================================================
   RECOMMENDATION SYSTEM — Pipeline Navigator
==============================================================

    1. Dataset Overview
    2. Build & Visualise Graph
    3. Basic Recommendations
    4. Full Model Pipeline + Evaluation
    5. Highlight Recommendation Paths
    6. Interactive Dynamic Input
    7. Advanced Graph + Scoreboard
    8. Run ALL steps (1-7) sequentially
    0. Exit

    Select step [1-7], 8 for all, 0 to exit:
```

Just type a number and press Enter — no need to type full file paths.

---

## System Pipeline (1–7)

The `pipeline/` folder contains a 7-section walkthrough for the recommendation engine. Use the **pipeline navigator** to run any section by number:

```bash
python run_pipeline.py
```

### What Each Section Does

| # | What It Does | Output |
|---|--------------|--------|
| **1** | Loads all 5 CSVs, prints node/edge counts, sample rows, and discovered categories | Console only |
| **2** | Builds heterogeneous graph (2,527 nodes, 10,417 edges), visualises a subset | `pipeline/2_graph.png` |
| **3** | Category-path and similar-user recommendations with dual-panel visualisation | `pipeline/3_basic_reco.png` |
| **4** | Runs all 3 models, hybrid blend, explanations, offline evaluation metrics | Console only |
| **5** | Traces colour-coded paths from user to each recommendation | `pipeline/5_paths.png` |
| **6** | **Interactive:** pick users, filter categories, compare two users side-by-side | `pipeline/6_*.png` |
| **7** | Multi-path graph + combined scoreboard + degree centrality analysis | `pipeline/7_advanced_graph.png` |

### Tips

- If the plot window blocks your terminal, set `MPLBACKEND=Agg` before running:
  ```powershell
  $env:MPLBACKEND="Agg"   # PowerShell
  export MPLBACKEND=Agg    # Bash
  ```
- Select **8** in the navigator to run all 7 sections sequentially.
- All PNG visualisations are saved in the `pipeline/` folder.
