# Hybrid Recommendation System — Workspace Instructions

## Project Overview

- **Language:** Python 3.12
- **Environment:** Virtual environment (`.venv`)
- **Package manager:** pip with pyproject.toml (src-layout)
- **Key libraries:** pandas, numpy, scikit-learn, networkx, scipy, matplotlib
- **Domain:** Cultural event recommendation system for Sri Lanka

## Architecture

Three recommendation models are blended by a hybrid ranker:
- **Knowledge-based** (`src/knowledge_based/`): Profile matching using art_interests, region_preference vs event art_forms, genres, region.
- **Graph-based** (`src/graph_based/`): Heterogeneous graph with PageRank + Jaccard/Adamic-Adar user similarity.
- **Trend-based** (`src/trend_based/`): Windowed attendance counts and growth rate scoring.
- **Hybrid** (`src/hybrid/`): Dynamic weight selection (cold_start / active / trending strategies), explanations.
- **Evaluation** (`src/evaluation/`): Precision@K, Recall@K, NDCG, MAP, coverage, diversity.

## Data Files (in `data/`)

| File | Rows | Key Columns |
|------|------|-------------|
| users.csv | 1,501 | user_id, name, art_interests, region_preference, culture_preferences |
| events.csv | 1,001 | event_id, name, art_forms, genres, region, ticket_price |
| artists.csv | 501 | artist_id, name, art_forms, genres, popularity |
| attends.csv | 7,197 | user_id, event_id, timestamp, rsvp_status, compatibility_score |
| follows.csv | 12,961 | user_id, artist_id, timestamp, compatibility_score |

**Important:** Do not add or modify CSV files. Use actual column names from the data (not generic names).

## Running

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .[dev]
python run_recommend.py        # Interactive menu
```

## System Pipeline (Steps 1–7)

The `pipeline/` folder contains a 7-step walkthrough for the recommendation engine:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `pipeline/step1_load_data.py` | Dataset overview: node/edge counts, samples |
| 2 | `pipeline/step2_build_graph.py` | Build and visualise graph structure |
| 3 | `pipeline/step3_basic_reco.py` | Category-path and similar-user recommendations |
| 4 | `pipeline/step4_run_model.py` | Full model pipeline + offline evaluation |
| 5 | `pipeline/step5_highlight_paths.py` | Colour-coded recommendation path traces |
| 6 | `pipeline/step6_dynamic_input.py` | Interactive: pick users, filter categories, compare |
| 7 | `pipeline/step7_advanced_graph.py` | Multi-path graph + scoreboard + centrality |

Step 4 requires `PYTHONPATH` set to the project root (or `pip install -e .`).

## Conventions

- List-like CSV columns (e.g. `['music', 'dance']`) are parsed with tokenized set-based matching.
- The `budget_col` is `None` by default since the data has no budget column.
- Git branch for improvements: `dev-improvs`.
