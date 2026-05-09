# Hybrid Cultural Event Recommendation System — Full Documentation

**Domain:** Cultural events in Sri Lanka  
**Language:** Python 3.12  
**Interface:** Streamlit dashboard (`app.py`) + CLI tools (`run_recommend.py`, `run_pipeline.py`)  
**Architecture:** Three independent models blended by a dynamic hybrid ranker

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Installation & Setup](#4-installation--setup)
5. [How to Run](#5-how-to-run)
6. [System Architecture](#6-system-architecture)
7. [Model 1 — Knowledge-Based Recommender](#7-model-1--knowledge-based-recommender)
8. [Model 2 — Graph-Based Recommender](#8-model-2--graph-based-recommender)
9. [Model 3 — Trend-Based Recommender](#9-model-3--trend-based-recommender)
10. [Hybrid Ranker](#10-hybrid-ranker)
11. [Explanations Engine](#11-explanations-engine)
12. [Output Format](#12-output-format)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Streamlit Dashboard (app.py)](#14-streamlit-dashboard-apppy)
15. [Pipeline Scripts](#15-pipeline-scripts)
16. [CLI Tools](#16-cli-tools)
17. [Artist Recommendations](#17-artist-recommendations)
18. [Cold-Start Behaviour](#18-cold-start-behaviour)
19. [Design Decisions & Trade-offs](#19-design-decisions--trade-offs)

---

## 1. Project Overview

This system recommends cultural events and artists to users based on a combination of three complementary signals:

| Signal | Source | Works without history? |
|--------|--------|----------------------|
| Profile match | User preferences vs event metadata | ✅ Yes |
| Social behaviour | What similar users attended / followed | ❌ Needs interactions |
| Trending momentum | Recent attendance growth across all users | ✅ Yes |

The three signals are blended into a single ranked list by the **HybridRanker**, which dynamically adjusts model weights based on how much interaction history a user has (cold-start vs active user).

---

## 2. Repository Structure

```
Recommendation/
├── app.py                        # Streamlit dashboard (6 pages)
├── run_recommend.py              # CLI interactive menu
├── run_pipeline.py               # Pipeline section launcher
├── pyproject.toml                # Package config and dependencies
│
├── data/
│   ├── users.csv                 # 200 users
│   ├── events.csv                # 150 events
│   ├── artists.csv               # 100 artists
│   ├── attends.csv               # 973 attendance records
│   └── follows.csv               # 1,684 follow relationships
│
├── src/
│   ├── knowledge_based/
│   │   └── knowledge_matcher.py  # KnowledgeMatcher class
│   ├── graph_based/
│   │   └── graph_similarity.py   # Jaccard + Adamic-Adar + event/artist recs
│   ├── trend_based/
│   │   └── trend_recommender.py  # TrendWindowRecommender class
│   ├── hybrid/
│   │   ├── recommend.py          # recommend_events() entry point
│   │   ├── hybrid_ranker.py      # HybridRanker + WeightScheme
│   │   └── explanations.py       # attach_explanations()
│   ├── evaluation/
│   │   └── metrics.py            # Precision, Recall, NDCG, MAP, Coverage, Diversity
│   ├── data_loader.py            # load_interactions() utility
│   └── data_preprocessing.py
│
├── pipeline/
│   ├── 1_load_data.py            # Dataset overview and node/edge counts
│   ├── 2_build_graph.py          # Graph construction and visualisation
│   ├── 3_basic_reco.py           # Category-path and similar-user recommendations
│   ├── 4_run_model.py            # Full model pipeline + offline evaluation
│   ├── 5_highlight_paths.py      # Colour-coded path traces
│   ├── 6_dynamic_input.py        # Interactive: pick users, compare
│   └── 7_advanced_graph.py       # Multi-path graph + centrality scoreboard
│
└── docs/
    ├── full_documentation.md     # This file
    └── system_guide.md
```

---

## 3. Dataset

All data is stored as CSV files in the `data/` directory. List-like columns (e.g. `['music', 'dance']`) are parsed with tokenized set-based matching throughout the codebase — no special loading step is required.

### users.csv — 200 rows

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `user_id` | str | `U0000` | Primary key |
| `name` | str | `Nuwan Wickremasinghe` | Display name |
| `ethnicity` | str | `sinhala` | |
| `language_preferences` | list str | `['sinhala', 'english']` | |
| `city` | str | `nuwara_eliya` | Used for location matching |
| `art_interests` | list str | `['music']` | **Primary matching field** — values: music, dance, drama, film |
| `culture_preferences` | list str | `['contemporary']` | Style preferences |
| `mood_preferences` | list str | `['patriotic', 'intense']` | |
| `activity_level` | str | `medium` | low / medium / high |
| `join_date` | datetime | `2025-05-13T...` | ISO 8601 |

### events.csv — 150 rows

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `event_id` | str | `E0000` | Primary key |
| `name` | str | `Celebration of Chathurika Wijesinghe` | |
| `artist_ids` | list str | `['A0036']` | Links to artists.csv |
| `art_forms` | list str | `['music']` | **Matched against user art_interests** |
| `genres` | list str | `['sad_songs']` | Also matched against art_interests |
| `language` | list str | `['english']` | |
| `city` | str | `batticaloa` | Used for location matching |
| `venue` | str | `Colombo University Arts Theatre` | |
| `style` | list str | `['contemporary']` | |
| `mood_tags` | list str | `['rebel', 'reflective']` | |
| `festival` | str | `esala_perahera` | Empty string if not a festival event |
| `festivals` | list str | `['esala_perahera']` | |
| `event_type` | str | `competition` | concert / workshop / festival / exhibition / performance / ritual_ceremony / competition |
| `date` | datetime | `2026-04-10T...` | ISO 8601 |
| `capacity` | int | `100` | |
| `ticket_price` | int | `3000` | In LKR; 0 = free |
| `status` | str | `upcoming` | |

### artists.csv — 100 rows

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `artist_id` | str | `A0000` | Primary key |
| `name` | str | `Chitrasena` | |
| `art_forms` | list str | `['film']` | Values: music / dance / drama / film |
| `genres` | list str | `['biographical', 'colonial_era']` | |
| `styles` | list str | `['historical_period']` | |
| `language` | list str | `['english']` | |
| `city` | str | `badulla` | |
| `style` | list str | `['contemporary']` | |
| `mood_tags` | list str | `['energetic', 'rebel', 'urban_street']` | |
| `festivals` | list str | `[]` | |
| `popularity` | str | `emerging` | emerging / mid_tier / established |
| `follower_count` | int | `1143` | |
| `verified` | bool | `False` | |

### attends.csv — 973 rows

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | str | Foreign key → users.csv |
| `event_id` | str | Foreign key → events.csv |
| `timestamp` | datetime | ISO 8601; parsed with `pd.to_datetime()` |
| `rsvp_status` | str | going / interested |
| `compatibility_score` | int | 1–10 |

### follows.csv — 1,684 rows

| Column | Type | Notes |
|--------|------|-------|
| `user_id` | str | Foreign key → users.csv |
| `artist_id` | str | Foreign key → artists.csv |
| `timestamp` | datetime | ISO 8601 |
| `compatibility_score` | int | 1–10 |

---

## 4. Installation & Setup

### Requirements

- Python 3.10 or newer (3.12 recommended)
- pip

### Steps

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/Recommendation.git
cd Recommendation

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows PowerShell)
.venv\Scripts\activate

# 4. Install package and dependencies
pip install -e .[dev]
```

### Dependencies (from pyproject.toml)

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, scoring DataFrames |
| `numpy` | Numerical operations in evaluation and visualisation |
| `scikit-learn` | General ML utilities |
| `networkx` | Graph construction and Adamic-Adar |
| `scipy` | Sparse matrix support |
| `streamlit` | Web dashboard (install separately) |
| `matplotlib` | Graph visualisation in pipeline and dashboard |
| `pytest` | Dev dependency for testing |

To install Streamlit:
```powershell
pip install streamlit matplotlib
```

---

## 5. How to Run

### Streamlit Dashboard

```powershell
.venv\Scripts\activate
streamlit run app.py
```

Opens at `http://localhost:8501`. The dashboard has 6 pages accessible from the sidebar.

### CLI Recommendation Menu

```powershell
python run_recommend.py
```

Presents 4 options:
1. Hybrid recommendations (all three models)
2. Trend-only (no user required)
3. Graph-only (similar users)
4. Hybrid with custom explanations

### Pipeline Navigator

```powershell
python run_pipeline.py
```

Presents a numbered menu to run any of the 7 pipeline sections interactively.

### Running Pipeline Sections Directly

```powershell
python pipeline/1_load_data.py
python pipeline/2_build_graph.py
# ... etc
```

> Section 4 (`4_run_model.py`) requires the package to be installed (`pip install -e .`) or `PYTHONPATH` set to the project root.

---

## 6. System Architecture

The recommendation pipeline has 5 sequential stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Input: user_id                            │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────┐
           ▼                       ▼                   ▼
  ┌────────────────┐    ┌────────────────────┐   ┌───────────────┐
  │  Knowledge-    │    │   Graph-Based      │   │  Trend-Based  │
  │  Based Model   │    │   Model            │   │  Model        │
  │  (profile)     │    │   (social)         │   │  (temporal)   │
  └───────┬────────┘    └────────┬───────────┘   └──────┬────────┘
          │                      │                       │
          ▼                      ▼                       ▼
   KnowledgeScore          GraphScore              TrendScore
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │      HybridRanker        │
                    │  (strategy-based blend)  │
                    └────────────┬────────────┘
                                 ▼
                          FinalScore
                                 │
                    ┌────────────▼────────────┐
                    │    attach_explanations   │
                    └────────────┬────────────┘
                                 ▼
                    Top-N events with Explanations
```

The entry point is `recommend_events()` in [src/hybrid/recommend.py](../src/hybrid/recommend.py). It orchestrates all three models, merges their scores into a single candidate DataFrame, calls the ranker, and attaches explanations.

---

## 7. Model 1 — Knowledge-Based Recommender

**File:** `src/knowledge_based/knowledge_matcher.py`  
**Class:** `KnowledgeMatcher`  
**Key method:** `recommend(user_id, top_n)`

### What it does

Scores every event purely from the user's profile — no past activity needed. Ideal for new users (cold-start).

### Fit

```python
km = KnowledgeMatcher(budget_col=None)
km.fit(users_df, events_df)
```

Validation checks that `user_id` column exists in users and `ticket_price` exists in events.

### Scoring Logic

For each event, `score_row()` accumulates from three sub-scores with default weights `(0.4, 0.3, 0.3)`:

```
KnowledgeScore = cat_score + loc_score + price_score
```

| Sub-score | Weight | Condition |
|-----------|--------|-----------|
| `cat_score` | **+0.4** | `user.art_interests` ∩ (`event.art_forms` ∪ `event.genres`) ≠ ∅ |
| `loc_score` | **+0.3** | `user.city` == `event.city` |
| `price_score` | **+0.3** | `event.ticket_price` ≤ `user.budget` — always 0 in production since `budget_col=None` |

### Token Parsing

All list-like column values (e.g. `"['music', 'drama']"`) are parsed by `_to_tokens()`:

```python
"['music', 'drama']"  →  {'music', 'drama'}
```

Handles plain strings, Python-list-formatted strings, and actual Python iterables.

### Possible KnowledgeScore Values

| Score | Meaning |
|-------|---------|
| **0.0** | No interest match, different city |
| **0.3** | Same city only |
| **0.4** | Interest match only |
| **0.7** | Interest match + same city |

### Fallback

If the user is not found in users.csv, all events are returned sorted by lowest `ticket_price` with `KnowledgeScore=0.0`.

---

## 8. Model 2 — Graph-Based Recommender

**File:** `src/graph_based/graph_similarity.py`  
**Key functions:** `jaccard_similar_users()`, `adamic_adar_similar_users()`, `recommend_from_similar_users()`

### What it does

Finds users who are behaviourally similar to the target and recommends events those users attended. Pure collaborative filtering — no event metadata used.

### Step 1: Build User Item Sets

```python
_build_user_item_sets(attends, follows)
```

Each user is represented as a **set of interaction tokens**:
- `"event:E0012"` — for each attended event
- `"artist:A0045"` — for each followed artist

This combined set (events + artists) is used for Jaccard similarity.

### Step 2: Jaccard Similarity

$$\text{Jaccard}(U, V) = \frac{|S_U \cap S_V|}{|S_U \cup S_V|}$$

Where $S_U$ = all events attended + artists followed by user U.

Returns a dict of `{other_user_id: similarity_score}`. Score of 0 means no overlap; 1.0 means identical interaction history.

### Step 3: Adamic-Adar Similarity

Builds a **bipartite graph** (`user:U → event:E` edges from attends.csv), projects it to a user-user graph via shared events, then computes Adamic-Adar:

$$\text{AA}(U, V) = \sum_{e \in N(U) \cap N(V)} \frac{1}{\log |N(e)|}$$

Where $N(e)$ = set of users who attended event $e$. Events attended by few people (niche events) contribute more to similarity than events attended by many.

### Step 4: Merge Similarities

```python
merged_score = 0.5 × Jaccard + 0.5 × AdamicAdar
```

The top 50 most similar users are selected.

### Step 5: Score Events

For each event attended by a similar user (that the target has **not** attended):

```
GraphScore[event] += similarity_score[that_user]
```

Events attended by multiple highly-similar users accumulate higher GraphScores.

### Fallback

Returns an empty DataFrame if the target user has no interactions at all. The hybrid ranker handles this gracefully by treating `GraphScore=0` for all events.

---

## 9. Model 3 — Trend-Based Recommender

**File:** `src/trend_based/trend_recommender.py`  
**Class:** `TrendWindowRecommender`  
**Key method:** `recommend(top_n, window_days)`

### What it does

Measures which events are gaining attendance momentum in a recent time window. No user profile needed — purely a global popularity signal.

### Fit

```python
trend = TrendWindowRecommender().fit(attends_df)
```

Parses timestamps with `pd.to_datetime(..., errors="coerce")` and drops rows with invalid timestamps.

### Time Windows

| Window | Definition |
|--------|-----------|
| `now_ts` | Max timestamp in attends.csv (most recent record) |
| Recent window | `[now_ts - 14 days, now_ts]` |
| Prior window | `[now_ts - 28 days, now_ts - 14 days]` |

Both windows default to 14 days but are independently configurable.

### Scoring Formula

For each event:

```
growth_rate = (recent_count - prev_count) / max(prev_count, 1)
raw_score   = recent_count + growth_rate
TrendScore  = (raw_score - min_raw) / (max_raw - min_raw)
```

- `recent_count` rewards events that are currently popular
- `growth_rate` rewards events that are **accelerating** even if their absolute count is smaller
- Min-max normalisation scales all trend scores to [0.0, 1.0]
- If all events have the same raw score, `TrendScore = 1.0` for all

### Output Columns

`event_id`, `recent_count`, `prev_count`, `growth_rate`, `TrendScore`

---

## 10. Hybrid Ranker

**File:** `src/hybrid/hybrid_ranker.py`  
**Class:** `HybridRanker`  
**Key method:** `rank(scores_df, user_interactions, top_n)`

### What it does

Blends the three model scores into a single `FinalScore` using strategy-based weights, then returns the top-N events.

### Weight Strategies

Defined as frozen dataclasses in `DEFAULT_WEIGHTS`:

| Strategy | alpha (Knowledge) | beta (Graph) | gamma (Trend) | When selected |
|----------|:-:|:-:|:-:|----------------|
| `cold_start` | **0.5** | 0.2 | 0.3 | `user_interactions < 5` |
| `active` | 0.2 | **0.5** | 0.3 | `user_interactions ≥ 5` |
| `trending` | 0.3 | 0.3 | **0.4** | Only if forced explicitly |

`interaction_threshold = 5` (configurable in constructor).

### FinalScore Formula

$$\text{FinalScore} = \alpha \times \text{KnowledgeScore} + \beta \times \text{GraphScore} + \gamma \times \text{TrendScore}$$

### Strategy Selection Logic

```python
def _choose_strategy(self, interactions, focus=None):
    if focus in self.weights:          # forced override
        return focus
    if interactions < 5:
        return "cold_start"
    return "active"
```

`trending` is never auto-selected — it must be explicitly passed as `focus="trending"`.

### Reasoning Behind Strategy Design

- **Cold-start users** have no reliable interaction history, so `GraphScore` would be near zero. Knowledge (profile matching) is weighted highest (0.5) since it is the only reliable signal.
- **Active users** have enough history for collaborative filtering to be meaningful, so `GraphScore` gets the highest weight (0.5).
- **Trend** always gets 0.3 in automatic modes — it acts as a tiebreaker and a signal for current relevance, but never dominates.

---

## 11. Explanations Engine

**File:** `src/hybrid/explanations.py`  
**Function:** `attach_explanations(recommendations, events, user_interests, user_city)`

### What it does

Appends a human-readable `Explanations` list to each row of the final ranked DataFrame.

### Logic Per Row

Three independent checks are applied in order; multiple reasons can fire for the same event:

| Explanation | Condition from code |
|-------------|---------------------|
| `"Matches your interests"` | `user.art_interests` ∩ (`event.art_forms` ∪ `event.genres`) ≠ ∅ |
| `"Popular among similar users"` | `row.GraphScore > 0` |
| `"Trending this week near you"` | `row.TrendScore > 0` AND `event.city == user.city` |
| `"Trending this week"` | `row.TrendScore > 0` AND city does not match |
| `"Recommended based on combined scores"` | Fallback — none of the above fired |

### Example Explanations Column Values

```python
["Matches your interests", "Popular among similar users"]
["Trending this week near you"]
["Matches your interests", "Trending this week"]
["Recommended based on combined scores"]
```

---

## 12. Output Format

The final output of `recommend_events()` is a pandas DataFrame with these columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `event_id` | str | — | e.g. `E0042` |
| `KnowledgeScore` | float | 0.0 – 0.7 | Profile match score |
| `GraphScore` | float | 0.0 – ∞ | Collaborative score (sum of user similarities) |
| `TrendScore` | float | 0.0 – 1.0 | Normalised trend momentum |
| `FinalScore` | float | 0.0 – ∞ | Weighted blend of all three |
| `Explanations` | list[str] | — | Human-readable reasons |

> **Note:** `GraphScore` is unbounded because it accumulates similarity scores from up to 50 users. In practice it is typically in the range [0, 5] for this dataset size.

---

## 13. Evaluation Metrics

**File:** `src/evaluation/metrics.py`  
**Entry point:** `evaluate(rec_map, rel_map, k, catalog, item_features)`

### Offline Evaluation Protocol

The evaluation uses **leave-one-out**: each user's chronologically last attended event is held out as ground truth. The model is trained on all prior attendances and asked to rank events — then checked whether the holdout event appears in the top-K recommendations.

### Metrics

#### Precision@K

$$\text{Precision@K} = \frac{\text{relevant items in top-K}}{K}$$

Answers: *Of the K events I showed, how many were actually relevant?*

#### Recall@K

$$\text{Recall@K} = \frac{\text{relevant items in top-K}}{|\text{all relevant items}|}$$

Answers: *Of all relevant events, how many did I catch in top-K?*

#### Average Precision (AP)

Precision computed at each relevant item's rank, then averaged:

$$\text{AP} = \frac{1}{\min(|R|, K)} \sum_{k=1}^{K} P(k) \cdot \text{rel}(k)$$

#### Mean Average Precision (MAP)

Average of AP across all users. Rewards placing relevant items higher in the list.

#### NDCG@K (Normalised Discounted Cumulative Gain)

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Where IDCG is the ideal ordering. Penalises relevant items found late in the list.

#### Coverage

$$\text{Coverage} = \frac{|\text{unique events recommended}|}{|\text{catalog}|}$$

Fraction of the total event catalog that gets recommended to at least one user. Low coverage means the system is too conservative.

#### Diversity

$$\text{Diversity} = \text{mean}_{(i,j)} \left(1 - \text{Jaccard}(\text{features}_i, \text{features}_j)\right)$$

Average pairwise dissimilarity of recommended events per user, based on art form features. Higher = more varied recommendations.

### evaluate() Return Value

```python
{
    "precision@k": float,
    "recall@k":    float,
    "map":         float,
    "ndcg":        float,
    "coverage":    float,
    "diversity":   float,
}
```

---

## 14. Streamlit Dashboard (app.py)

Run with: `streamlit run app.py`

The dashboard has 6 pages selectable from the sidebar:

### Page 1 — 🏠 Dashboard

Overview of the selected user's profile. Shows:
- User profile card (art interests, city, culture preferences)
- Event interaction graph — visualises which events the user attended and the art form categories they belong to
- Artist interaction graph — visualises which artists they follow and matching interest categories

### Page 2 — 🎭 Event Recommendations

Runs the full hybrid pipeline (`run_recommendation_pipeline()`) for the selected user. Shows:
- Model strategy used (`cold_start` or `active`) and interaction count
- Recommended events with scores (`KnowledgeScore`, `GraphScore`, `TrendScore`, `FinalScore`, `Explanations`)
- Recommendation path visualisation (two graph panels: category-path logic + collaborative filtering logic)
- Score distribution bar charts per model
- Similar users table with shared interest analysis
- User comparison tool (with interaction path graph)

### Page 3 — 🎨 Artist Discovery

Runs the artist recommendation pipeline (`run_artist_pipeline()`) for the selected user. Shows:
- Recommended artists with scores and art forms
- Artist interaction path graph (User → Interest Categories → Artists)
- Similar users comparison for artist tastes

### Page 4 — 📊 Model Comparison

Side-by-side comparison of multiple users' recommendations. Allows:
- Selecting up to 4 users to compare
- Viewing ranked results from all three models independently for each user
- Comparing `FinalScore` rankings across users

### Page 5 — 📈 Evaluation Metrics

Offline evaluation results. Shows:
- Precision@K, Recall@K, NDCG, MAP as metric cards
- Coverage and Diversity
- Per-user breakdown table (train event count, whether holdout was in top-K)
- Score distribution histogram

### Page 6 — 📈 Data Explorer

Raw data browser with 5 tabs:
- **Users** — city distribution bar chart + user table
- **Events** — event type distribution + event table (columns: event_id, name, city, art_forms, genres, event_type, date, ticket_price, status)
- **Artists** — popularity distribution + artist table (columns: artist_id, name, art_forms, genres, city, popularity, follower_count, verified)
- **Attends** — attendance over time line chart + attendance table
- **Follows** — top 15 most-followed artists bar chart + follows table

### Data Loading

All data is loaded once via `@st.cache_data`:

```python
@st.cache_data(show_spinner=False)
def load_data():
    users   = pd.read_csv(DATA_DIR / "users.csv")
    events  = pd.read_csv(DATA_DIR / "events.csv")
    artists = pd.read_csv(DATA_DIR / "artists.csv")
    attends = pd.read_csv(DATA_DIR / "attends.csv")
    follows = pd.read_csv(DATA_DIR / "follows.csv")
    attends["timestamp"] = pd.to_datetime(attends["timestamp"])
    return users, events, artists, attends, follows
```

---

## 15. Pipeline Scripts

The `pipeline/` folder contains 7 standalone scripts that walk through the system step-by-step. They are designed for explainability and educational walkthroughs.

### 1 — `1_load_data.py`: Dataset Overview

Loads all 5 CSV files and prints:
- Node counts (users, events, artists, categories)
- Edge counts (attends, follows, event→artist from `artist_ids` column)
- Sample rows from each file
- All discovered category/genre tokens

### 2 — `2_build_graph.py`: Build & Visualise Graph

Constructs the heterogeneous graph:
- Nodes: users (blue), events (amber), categories/genres (green)
- Edges: user→event (attended), event→category (belongs_to), user→category (intersect)

Visualises a small subset of nodes using `networkx` + `matplotlib`. Explains what each node and edge type represents.

### 3 — `3_basic_reco.py`: Category-Path and Similar-User Recommendations

Demonstrates two simple recommendation strategies without the full hybrid pipeline:
1. **Category-Path** — find user's attended events → extract categories → find new events sharing those categories
2. **Similar-User** — find users who attended the same events → recommend events they attended that the target has not

Visualises both strategies as graph paths using colour-coded nodes.

### 4 — `4_run_model.py`: Full Model Pipeline + Offline Evaluation

Runs the complete hybrid pipeline including all three models, hybrid ranking, and offline evaluation. Requires `pip install -e .` or PYTHONPATH set. Prints per-model scores and final ranked output.

### 5 — `5_highlight_paths.py`: Colour-Coded Path Traces

Highlights the specific paths through the graph that led to each top recommendation. Different colours for knowledge-path, graph-path, and trend-path recommendations.

### 6 — `6_dynamic_input.py`: Interactive: Pick Users, Filter, Compare

Interactive session where you can:
- Select any user by ID from a displayed list
- Filter recommendations by art form category
- Compare two users side by side
- Generate event and artist recommendations interactively

### 7 — `7_advanced_graph.py`: Multi-Path Graph + Centrality Scoreboard

Advanced visualisation with two panels:
- **Panel 1** — multi-path graph showing all recommendation paths simultaneously with a scoreboard table overlay
- **Panel 2** — centrality analysis (degree and betweenness) to identify the most influential nodes in the graph

---

## 16. CLI Tools

### run_recommend.py

```
python run_recommend.py
```

**Menu options:**

| Option | Function | Description |
|--------|----------|-------------|
| 1 | `run_hybrid()` | Full hybrid: all 3 models + weighted blend |
| 2 | `run_trend_only()` | Trend only, no user ID needed |
| 3 | `run_graph_only()` | Graph (collaborative) only |
| 4 | `run_with_explanations()` | Hybrid with custom user interests override |
| 0 | Exit | |

**Option 4 — Custom Explanations:**  
Accepts `user_id`, `top_n`, manually typed interests (comma-separated), and city override. Useful for testing "what if" scenarios — e.g. what would be recommended if this user's interests were different.

### run_pipeline.py

```
python run_pipeline.py
```

Displays a menu of all 7 pipeline sections and runs the selected one as a subprocess.

---

## 17. Artist Recommendations

Artist recommendations mirror the event recommendation flow but use artists.csv and follows.csv instead of events.csv and attends.csv.

### Profile Match (Knowledge-style)

For each artist not already followed:
```
art_cats = tokens(artist.art_forms) ∪ tokens(artist.genres)
score = |user.art_interests ∩ art_cats| / |user.art_interests ∪ art_cats|
```
Jaccard overlap between user interests and artist art forms/genres.

### Graph Match (Collaborative)

Uses `recommend_artists_from_similar_users()` — same Jaccard + Adamic-Adar similarity pipeline, but scores artists followed by similar users (excluding already-followed artists).

### Hybrid Artist Score

```
FinalScore = KnowledgeScore + GraphScore
```

Sorted descending. Explanations attached via same `attach_explanations()` mechanism.

---

## 18. Cold-Start Behaviour

The system handles three tiers of new-user situations:

| Situation | `KnowledgeScore` | `GraphScore` | `TrendScore` | Strategy |
|-----------|:-:|:-:|:-:|---------|
| Brand new user, no profile | 0.0 (fallback: sort by price) | 0.0 | ✅ Active | — |
| User has profile, 0 attendances | ✅ Active | 0.0 | ✅ Active | `cold_start` (α=0.5) |
| User has 1–4 attendances | ✅ Active | ~Low | ✅ Active | `cold_start` (α=0.5) |
| User has 5+ attendances | ✅ Active | ✅ Active | ✅ Active | `active` (β=0.5) |

The `cold_start` strategy (α=0.5, β=0.2, γ=0.3) ensures that even with no history, the system returns meaningful results via profile matching and trending events rather than recommending nothing.

---

## 19. Design Decisions & Trade-offs

### Why three separate models?

Each model covers a different failure case of the others:

| Model | Fails when… | Backed up by… |
|-------|-------------|---------------|
| Knowledge | User has unusual/sparse interests | Graph + Trend |
| Graph | User is new (cold-start) | Knowledge + Trend |
| Trend | All events are equally popular | Knowledge + Graph |

### Why Jaccard + Adamic-Adar together?

- **Jaccard** is simple and interpretable but gives equal weight to all shared items
- **Adamic-Adar** gives higher weight to rare shared items (niche events signal stronger affinity than blockbuster events)
- Equal blend (50/50 default `alpha=0.5`) balances global overlap with niche signal

### Why min-max normalise TrendScore but not GraphScore?

- `TrendScore` is always normalised [0, 1] so it can be compared across different time windows and dataset sizes
- `GraphScore` is left in natural units (sum of similarity scores) because normalising it would lose the absolute magnitude — a low raw GraphScore means the user genuinely has few similar peers, which is meaningful information for the ranker

### Why `budget_col=None` by default?

The dataset's `ticket_price` column represents event cost, but there is no `budget` column in users.csv. Setting `budget_col=None` disables the price component of `KnowledgeScore` cleanly without raising errors. The price column is still used for **sorting** events when multiple have equal scores.

### List-like column parsing

CSV columns like `"['music', 'film']"` are stored as Python-repr strings. The `_to_tokens()` function in each module strips the brackets and quotes and splits on commas — a deliberate choice to keep the CSV files human-readable without requiring a custom loader or JSON parsing.
