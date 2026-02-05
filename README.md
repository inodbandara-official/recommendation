# Hybrid Recommendation System

This project scaffolds a hybrid recommendation system combining knowledge-based, graph-based, and trend-based approaches. Data is expected as CSV files under the `data/` directory.

## Prerequisites
- Python 3.10+
- pip (or another PEP 517 compatible installer)

## Setup
1. Create and activate a virtual environment (recommended).
2. Install the project in editable mode:
   ```bash
   pip install -e .[dev]
   ```

## Project Layout
- `data/`: place source CSV files here (e.g., interactions, items, users).
- `src/`: Python source (packaged with src-layout).
  - `knowledge_based/`: rule and constraint-driven recommenders.
  - `graph_based/`: network-based recommendation logic.
  - `trend_based/`: popularity/time-based recommenders.
  - `hybrid/`: ensemble and blending utilities.
  - `data_loader.py`: helpers to load and validate CSV input.
- `notebooks/`: exploratory analysis and experiments.

## Development Notes
- Stubs illustrate expected interfaces: `fit(...)` to ingest data, `recommend(user_id, top_k)` to produce ranked items.
- Extend or replace stubs with domain logic; add tests alongside new modules.
- Update dependencies in `pyproject.toml` if you introduce new libraries.

## Running Recommendations

1. Ensure data CSVs are in `data/` (`users.csv`, `events.csv`, `attends.csv`, `follows.csv`, `artists.csv`). Cleaned versions (`cleaned_*.csv`) are used automatically if present.

2. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   ```
   
   Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\activate
     ```
   - **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
   
   Install the project:
   ```bash
   pip install -e .[dev]
   ```

3. Run the recommendation system:
   ```bash
   python run_recommend.py
   ```

4. Select an option from the menu:
   ```
   === Hybrid Recommendation System ===
   1. Hybrid recommendations (knowledge + graph + trend)
   2. Trend-only recommendations
   3. Graph-only recommendations (similar users)
   4. Hybrid with custom explanations
   0. Exit
   ```

### Menu Options

| Option | Description |
|--------|-------------|
| **1** | Full hybrid blend combining knowledge-based, graph-based, and trend-based scores with auto-selected weights based on user activity level. |
| **2** | Trend-only: shows trending events based on recent attendance and growth rate. No user required. |
| **3** | Graph-only: recommends events attended by similar users (Jaccard + Adamic-Adar similarity). |
| **4** | Hybrid with custom explanations: lets you specify interests and region for tailored explanation text. |

### Example Session
```
python run_recommend.py

=== Hybrid Recommendation System ===
1. Hybrid recommendations (knowledge + graph + trend)
...
Select an option [1-4, 0 to exit]: 1
Enter user_id (e.g. U0008): U0006
Enter top_n (default 10): 10

Generating top 10 hybrid recommendations for U0006...

  event_id  KnowledgeScore  GraphScore  TrendScore  FinalScore  Explanations
0    E0123            0.70        0.45        0.60        0.58  [Matches your interests, Trending this week]
...
```
