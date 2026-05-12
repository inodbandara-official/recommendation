# Hybrid Recommendation System

A hybrid cultural event recommendation system for Sri Lanka, combining **knowledge-based**, **graph-based**, and **trend-based** approaches with strategy-aware blending and offline evaluation.

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

### Step 5 — Verify the Dataset

The system reads a **single JSON dataset**. Make sure this file exists:

```
data/rasaswadaya_large_dataset.json
```

It contains:

| Section | Count |
| --- | --- |
| `users` | 10,000 |
| `artists` | 500 |
| `events` | 2,000 |
| `interactions.follows` | 108,600 |
| `interactions.attends` | 39,986 |

All loading is centralised in [`src/data_io.py`](src/data_io.py) — every module (hybrid recommender, pipeline scripts, Streamlit app) reads through it. There are no CSVs in `data/` anymore.

### Step 6 — Run the System

You have **three entry points**:

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
```

#### Option C: Streamlit UI

```bash
.\.venv\Scripts\activate; streamlit run app.py
```

#### Option D: Offline Evaluation

```bash
# Baseline metrics (precision/recall/MAP/NDCG/coverage/diversity)
python pipeline/8_evaluate.py

# Grid search over weights + trend window
python pipeline/8_evaluate.py --grid

# Compare graph backends
python pipeline/8_evaluate.py --graph-mode pagerank
```

Reports land in [`reports/`](reports/) as JSON.

---

## How the Hybrid Recommender Works

For each user the system computes three component scores and blends them with strategy-aware weights:

| Component | Source | Captures |
| --- | --- | --- |
| **KnowledgeScore** | [`src/knowledge_based/knowledge_matcher.py`](src/knowledge_based/knowledge_matcher.py) | Graded Jaccard overlap between user profile (`art_interests`, `mood_preferences`, `culture_preferences`, `language_preferences`, `city`) and event metadata, plus an `activity_level`-derived budget proxy and a popularity prior from event capacity. Vectorised — events are pre-tokenised once at fit time. |
| **GraphScore** | [`src/graph_based/graph_similarity.py`](src/graph_based/graph_similarity.py) (default) or [`graph_builder.py`](src/graph_based/graph_builder.py) (`--graph-mode pagerank`) | Cosine similarity over a sparse user × (event ∪ artist) matrix combined with Adamic-Adar weighting; or personalized PageRank over the user-event-artist heterogeneous graph. |
| **TrendScore** | [`src/trend_based/trend_recommender.py`](src/trend_based/trend_recommender.py) | Recent attendance count + growth rate, min-max normalised. Window length is tunable. |

The [`HybridRanker`](src/hybrid/hybrid_ranker.py) min-max normalises each component to `[0, 1]` before applying the strategy weights, so no single recommender can dominate by virtue of its raw score scale.

### Strategy gate

The ranker picks one of three weight schemes per user:

| Strategy | Trigger | Default `(α, β, γ)` for (Knowledge, Graph, Trend) |
| --- | --- | --- |
| `cold_start` | `activity_level == "low"` *or* train interactions `< interaction_threshold` | `(0.50, 0.20, 0.30)` |
| `active` | `activity_level == "high"` *or* enough train interactions | `(0.20, 0.50, 0.30)` |
| `trending` | Explicit override only | `(0.30, 0.30, 0.40)` |

These can be overridden per-call via the `weights=` argument to `recommend_events`.

### Future-event filter

`recommend_events(future_only=True)` drops events whose `date` is before `today` so recommendations never include events that have already happened.

---

## System Pipeline (1–8)

The `pipeline/` folder contains step scripts. Use the **pipeline navigator** to run any section by number:

```bash
python run_pipeline.py
```

### What Each Section Does

| # | What It Does | Output |
|---|--------------|--------|
| **1** | Loads the JSON dataset, prints node/edge counts, sample rows, and discovered categories | Console only |
| **2** | Builds heterogeneous graph, visualises a subset | `pipeline/2_graph.png` |
| **3** | Category-path and similar-user recommendations with dual-panel visualisation | `pipeline/3_basic_reco.png` |
| **4** | Runs all 3 models, hybrid blend, explanations, offline evaluation metrics | Console only |
| **5** | Traces colour-coded paths from user to each recommendation | `pipeline/5_paths.png` |
| **6** | **Interactive:** pick users, filter categories, compare two users side-by-side | `pipeline/6_*.png` |
| **7** | Multi-path graph + combined scoreboard + degree centrality analysis | `pipeline/7_advanced_graph.png` |
| **8** | Offline evaluation harness — temporal holdout, metrics, optional grid search | `reports/metrics.json`, `reports/grid_search.json` |

### Tips

- If the plot window blocks your terminal, set `MPLBACKEND=Agg` before running:
  ```powershell
  $env:MPLBACKEND="Agg"   # PowerShell
  export MPLBACKEND=Agg    # Bash
  ```
- Select **8** in the navigator to run all 7 visual sections sequentially. The eval script (`pipeline/8_evaluate.py`) is run directly, not from the navigator.
- All PNG visualisations are saved in the `pipeline/` folder; metrics JSON in `reports/`.

---

## Evaluation

The eval harness ([`pipeline/8_evaluate.py`](pipeline/8_evaluate.py)) does a **temporal holdout**: for each user with multi-attend history, the most recent 20% of their attends becomes the test set; the rest is used to compute recommendations. Metrics:

- `precision@k`, `recall@k`, `map`, `ndcg` — ranking quality vs. ground-truth holdout
- `coverage` — fraction of the catalog actually recommended
- `diversity` — average pairwise dissimilarity (1 − Jaccard) of recommended events on `(art_forms, genres, moods)`

### Tuned defaults (top-K = 10, 1,000 users sampled)

After two grid searches (one tuning the `active` weight scheme, one tuning `cold_start`) the live `HybridRanker` ships these defaults:

| Strategy | Best `(α, β, γ)` |
| --- | --- |
| `cold_start` | `(0.25, 0.50, 0.25)` |
| `active` | `(0.00, 0.75, 0.25)` |
| `trending` | `(0.30, 0.30, 0.40)` (untuned) |
| `trend_window_days` | `60` |

Final evaluation on 1,000 users at these defaults:

| Metric | Value |
| --- | --- |
| **Hit-Rate@10** *(% of users with ≥ 1 relevant item in top-10 — closest to "accuracy")* | **11.9%** |
| **Recall@10** | **10.5%** |
| **Precision@10** | 1.2% |
| **NDCG@10** | 0.055 |
| **MAP** | 0.037 |
| **MRR@10** | 0.043 |
| **Coverage** | 80.7% |
| **Diversity** | 0.89 |

**Reading the numbers.** Recommendation is ranked retrieval, not classification — there is no single "accuracy %". `Hit-Rate@10 = 11.9%` is the closest single number ("about 1 in 8 users got at least one useful recommendation in their top 10"). The bigger story is that diagnostic correlation analysis (`reports/diagnostics.json`) shows the synthetic dataset's stated user preferences only weakly predict actual attendance (Pearson ≈ +0.02). On real production data with richer behavioural signals these numbers should be substantially higher.

### Diagnostics

The eval harness also writes `reports/diagnostics.json` — per-user correlation between each component score and held-out attendance. Use it to decide whether weak metrics are caused by the model or by the data.

---

## Project Layout

```text
Recommendation/
├── data/
│   └── rasaswadaya_large_dataset.json    # single source of truth
├── src/
│   ├── data_io.py                        # JSON loader (cached)
│   ├── data_inspection.py
│   ├── knowledge_based/
│   │   └── knowledge_matcher.py          # vectorised, multi-field, graded
│   ├── graph_based/
│   │   ├── graph_similarity.py           # sparse cosine + Adamic-Adar
│   │   └── graph_builder.py              # personalized PageRank
│   ├── trend_based/
│   │   └── trend_recommender.py
│   ├── hybrid/
│   │   ├── recommend.py                  # main entry point
│   │   ├── hybrid_ranker.py              # strategy-aware weighting
│   │   └── explanations.py
│   └── evaluation/
│       └── metrics.py
├── pipeline/                             # 1..7 visual demos, 8 evaluation
├── reports/                              # generated metrics + grid results
├── app.py                                # Streamlit dashboard
├── run_recommend.py                      # interactive menu
└── run_pipeline.py                       # pipeline navigator
```

---

## Programmatic Use

```python
from pathlib import Path
import pandas as pd
from src.hybrid import recommend_events

recs = recommend_events(
    user_id="U00035",
    top_n=10,
    data_dir=Path("data"),
    graph_mode="similarity",         # or "pagerank"
    trend_window_days=60,
    today=pd.Timestamp("2026-05-09"),
    future_only=True,
)
print(recs[["event_id", "FinalScore", "Explanations"]])
```
