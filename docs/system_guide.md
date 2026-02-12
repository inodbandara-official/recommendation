# Hybrid Recommendation System — Detailed Guide

## Table of Contents

1. [Overview](#overview)
2. [Hybrid Recommendation System (Interactive Menu)](#hybrid-recommendation-system-interactive-menu)
   - [Option 1: Hybrid Recommendations](#option-1-hybrid-recommendations)
   - [Option 2: Trend-Only Recommendations](#option-2-trend-only-recommendations)
   - [Option 3: Graph-Only Recommendations](#option-3-graph-only-recommendations)
   - [Option 4: Hybrid with Custom Explanations](#option-4-hybrid-with-custom-explanations)
3. [Recommendation System — Pipeline Navigator](#recommendation-system--pipeline-navigator)
   - [1: Dataset Overview](#1-dataset-overview)
   - [2: Build & Visualise Graph](#2-build--visualise-graph)
   - [3: Basic Recommendation Analysis](#3-basic-recommendation-analysis)
   - [4: Full Model Pipeline + Evaluation](#4-full-model-pipeline--evaluation)
   - [5: Highlight Recommendation Paths](#5-highlight-recommendation-paths)
   - [6: Interactive Dynamic Input](#6-interactive-dynamic-input)
   - [7: Advanced Graph + Scoreboard](#7-advanced-graph--scoreboard)

---

## Overview

This system recommends cultural events in Sri Lanka by combining three independent recommendation models into a single hybrid output. The dataset contains **150 users**, **120 events**, **60 artists**, **731 attendance records**, and **1,238 follow relationships**.

The system can be used in two ways:

| Entry Point | Command | Purpose |
|---|---|---|
| Interactive Menu | `python run_recommend.py` | Directly generate recommendations for a given user |
| Pipeline Navigator | `python run_pipeline.py` | Walk through the full system pipeline section-by-section |

---

## Hybrid Recommendation System (Interactive Menu)

**Command:** `python run_recommend.py`

The interactive menu exposes four distinct recommendation modes. Each mode prompts for input (user ID, number of results, etc.) and prints recommendations to the console.

---

### Option 1: Hybrid Recommendations

**What it does:** Generates a blended recommendation by running all three models (knowledge-based, graph-based, and trend-based) and combining their scores with dynamically chosen weights.

**How it works:**

1. **Knowledge-Based Scoring** — The `KnowledgeMatcher` compares the target user's profile (`art_interests`, `city`) against every event's metadata (`art_forms`, `genres`, `city`, `ticket_price`). It uses tokenized set-intersection to measure category overlap and city match, producing a `KnowledgeScore` between 0 and 1 for each event. This model requires **no past activity**, making it effective for cold-start users.

2. **Graph-Based Scoring** — The system identifies users who are behaviourally similar to the target user using two complementary metrics:
   - **Jaccard Similarity** — Treats each user as a set of attended events and followed artists, then computes the Jaccard index (intersection over union) between the target user and every other user.
   - **Adamic-Adar Index** — Constructs a bipartite user–event graph, projects it onto user-only space, and computes Adamic-Adar scores that give higher weight to shared events with fewer total attendees (i.e., niche shared interests are more informative).
   
   These two scores are merged (default 50/50 blend). Events attended by the most similar users — but not yet attended by the target — are surfaced with a `GraphScore`.

3. **Trend-Based Scoring** — The `TrendWindowRecommender` analyses attendance timestamps over a configurable time window (default: 14 days). It counts recent attendances and compares them against a prior window of equal length to compute a growth rate. The final `TrendScore` combines raw recent popularity with the growth rate, normalized to [0, 1]. This captures events that are currently gaining momentum, not just historically popular ones.

4. **Hybrid Ranking** — The `HybridRanker` selects a weight strategy based on the user's activity level:

   | Strategy | Condition | Knowledge (α) | Graph (β) | Trend (γ) |
   |---|---|---|---|---|
   | `cold_start` | < 5 interactions | 0.5 | 0.2 | 0.3 |
   | `active` | ≥ 5 interactions | 0.2 | 0.5 | 0.3 |
   | `trending` | Manually forced | 0.3 | 0.3 | 0.4 |

   The final score is: `FinalScore = α × KnowledgeScore + β × GraphScore + γ × TrendScore`

   Cold-start users lean on profile matching (knowledge), while active users lean on collaborative signals (graph). The top N events are returned sorted by `FinalScore`.

5. **Explanations** — Each recommended event is annotated with a human-readable explanation such as "Matches your interests", "Popular among similar users", or "Trending this week near you".

**Output columns:** `event_id`, `KnowledgeScore`, `GraphScore`, `TrendScore`, `FinalScore`, `Explanations`

---

### Option 2: Trend-Only Recommendations

**What it does:** Returns the top trending events across the entire platform, independent of any specific user.

**How it works:** Uses the `TrendWindowRecommender` to analyse recent attendance patterns. The model divides the timeline into a recent window and a prior comparison window (both configurable), counts event attendances in each, and computes a growth rate. Events are scored by combining recent popularity with their growth trajectory.

**When to use:** Useful for homepage recommendations, editorial "What's Hot" sections, or when no user identity is available. No user ID is required — this is a purely global signal.

**Input:** Number of results (`top_n`) and window length in days (`window_days`).

**Output columns:** `event_id`, `recent_count`, `prev_count`, `growth_rate`, `TrendScore`

---

### Option 3: Graph-Only Recommendations

**What it does:** Recommends events based solely on collaborative filtering through user similarity in the interaction graph.

**How it works:** Computes Jaccard and Adamic-Adar similarity between the target user and all other users based on shared attendance and follow behaviour. It then aggregates the events attended by the most similar users (weighted by similarity score), excluding events the target has already attended. The result is a ranked list of events that similar users enjoyed.

**When to use:** Best for active users with a meaningful interaction history. Shows the pure collaborative signal without knowledge or trend adjustments. Useful for debugging or understanding how much the graph model contributes to the hybrid output.

**Input:** User ID (`user_id`) and number of results (`top_n`).

**Output columns:** `event_id`, `GraphScore`

---

### Option 4: Hybrid with Custom Explanations

**What it does:** Runs the full hybrid pipeline (same as Option 1), then generates enriched, context-aware explanations using **user-provided interests and city**.

**How it works:** After computing the hybrid scores, the `attach_explanations()` function cross-references each recommended event's metadata (art forms, genres, city) against the interests and city you provide at the prompt. This produces more personalised explanations than Option 1, where explanations are derived only from the stored user profile.

**Explanation types generated:**

| Explanation | Condition |
|---|---|
| "Matches your interests" | Event art forms/genres overlap with the interests you entered |
| "Popular among similar users" | Event has a non-zero GraphScore |
| "Trending this week near you" | Event has a non-zero TrendScore and is in your city |
| "Trending this week" | Event has a non-zero TrendScore but is in a different city |

**When to use:** When you want to simulate how the system would explain recommendations for a user with specific interests, or when presenting the system to demonstrate its explainability capabilities.

**Input:** User ID, number of results, comma-separated interests (e.g., `music,dance`), and city (e.g., `Colombo`).

**Output columns:** `event_id`, `FinalScore`, `Explanations`

---

## Recommendation System — Pipeline Navigator

**Command:** `python run_pipeline.py`

The pipeline navigator provides a structured 7-section walkthrough of the entire recommendation pipeline. Each section is a self-contained Python script in the `pipeline/` folder that can be run individually or sequentially. The navigator sets up the correct Python path and environment automatically.

---

### 1: Dataset Overview

**Script:** `pipeline/1_load_data.py`

Loads all five CSV datasets (users, events, artists, attends, follows) and prints a comprehensive overview of the data:

- **Node counts** — Number of unique users, events, and artists in the system.
- **Edge counts** — Number of attendance records (user→event links) and follow relationships (user→artist links).
- **Sample rows** — Displays the first few rows of each dataset so you can see the structure and column types.
- **Category discovery** — Extracts and lists all unique art forms, genres, and cities found across users, events, and artists. This shows the cultural taxonomy the system works with.

**Purpose:** Establishes confidence that the data is loaded correctly and gives the audience a clear picture of the dataset's scale and structure before any processing begins.

**Output:** Console only (no files saved).

---

### 2: Build & Visualise Graph

**Script:** `pipeline/2_build_graph.py`

Constructs the heterogeneous graph that powers the graph-based recommender:

- **Graph construction** — Builds a NetworkX graph with three node types (users, events, artists) and two edge types (attends, follows). The full graph contains approximately **2,527 nodes** and **10,417 edges**.
- **Subgraph sampling** — Selects a manageable subset of nodes for visualisation so the graph layout is readable.
- **Colour-coded visualisation** — Renders users, events, and artists in distinct colours with a legend. Edge types (attends vs. follows) are also visually distinguished.
- **Graph statistics** — Prints degree distribution, connected components, and density metrics.

**Purpose:** Makes the abstract graph structure tangible. Demonstrates that the system models relationships between users, events, and artists as a connected network — the foundation for collaborative filtering.

**Output:** Saves `pipeline/2_graph.png`.

---

### 3: Basic Recommendation Analysis

**Script:** `pipeline/3_basic_reco.py`

Generates two types of baseline recommendations and visualises the recommendation paths:

- **Category-path recommendations** — Finds events that match a sample user's art interests by tracing paths through the graph: User → Interest Category → Events tagged with that category.
- **Similar-user recommendations** — Identifies users with overlapping attendance/follow patterns and recommends what those similar users attended.
- **Dual-panel graph visualisation** — Renders both recommendation approaches side-by-side, with highlighted edges showing how each recommendation was reached.

**Purpose:** Shows the two fundamental recommendation strategies (content-based and collaborative) before the hybrid model is introduced. Provides visual intuition for how graph traversal produces recommendations.

**Output:** Saves `pipeline/3_basic_reco.png`.

---

### 4: Full Model Pipeline + Evaluation

**Script:** `pipeline/4_run_model.py`

Runs the complete hybrid recommendation pipeline and evaluates it with offline metrics:

- **Model execution** — Runs all three models (knowledge, graph, trend) for a sample of 30 users, blends them with the `HybridRanker`, and attaches explanations.
- **Sample output** — Prints the top recommendations with all score columns and explanations for a few representative users.
- **Offline evaluation** — Computes standard information retrieval metrics across the user sample:
  - **Precision@K** — Fraction of top-K recommendations that are relevant (user actually attended).
  - **Recall@K** — Fraction of relevant events that appear in the top-K.
  - **NDCG** (Normalized Discounted Cumulative Gain) — Measures ranking quality, giving more credit to relevant items ranked higher.
  - **MAP** (Mean Average Precision) — Average precision across all relevant items, averaged over users.
  - **Coverage** — Proportion of the total event catalogue that appears in at least one user's recommendations.
  - **Diversity** — Average pairwise dissimilarity among recommended events (based on genre/category features).

**Purpose:** Proves the system works end-to-end and provides quantitative evidence of recommendation quality. This is the core technical validation section.

**Output:** Console only (metrics printed as a summary table).

---

### 5: Highlight Recommendation Paths

**Script:** `pipeline/5_highlight_paths.py`

Traces and visualises the specific paths through the graph that connect a user to each recommended event:

- **Path tracing** — For each recommendation, finds the shortest path(s) in the heterogeneous graph from the target user to the recommended event.
- **Colour-coded paths** — Each recommendation gets a unique colour. The paths are overlaid on the graph so you can see exactly which intermediate nodes (shared artists, similar users, common categories) connect the user to the event.
- **Legend and annotations** — Each path is labelled with the event name and the reason for the recommendation.

**Purpose:** Provides explainability at the graph level. Instead of just showing a score, this section visually answers "why was this event recommended?" by showing the structural connections. Particularly effective for communicating the value of graph-based recommendations.

**Output:** Saves `pipeline/5_paths.png`.

---

### 6: Interactive Dynamic Input

**Script:** `pipeline/6_dynamic_input.py`

An interactive mode where the presenter can explore the system live:

- **User selection** — Choose any user ID from the dataset (with autocomplete suggestions).
- **Category filtering** — Filter recommendations to specific art forms or genres (e.g., show only music events, or only dance events).
- **Side-by-side comparison** — Select two users and compare their recommendations next to each other, highlighting where they overlap and diverge.
- **Live visualisation** — Each query generates a fresh graph visualisation with the selected user's recommendation paths highlighted.

**Purpose:** Turns the system into a live, interactive tool during presentation. The panel can request specific scenarios ("What would you recommend for a user in the Western province who likes traditional dance?") and see results immediately.

**Output:** Saves visualisations as `pipeline/6_*.png` (multiple files depending on interactions).

---

### 7: Advanced Graph + Scoreboard

**Script:** `pipeline/7_advanced_graph.py`

The final capstone section combining advanced graph analysis with a consolidated scoreboard:

- **Multi-path graph** — Visualises multiple recommendation paths simultaneously on a single graph, showing how different models (knowledge, graph, trend) each contribute paths to the same set of recommendations.
- **Combined scoreboard** — Displays a ranked table of recommended events with all three model scores, the final blended score, and the weight strategy used. Presented as a formatted console table.
- **Degree centrality analysis** — Identifies the most connected nodes in the graph (most popular events, most active users, most followed artists) using degree centrality. Highlights hub nodes that act as bridges in the recommendation network.

**Purpose:** Demonstrates the full sophistication of the system — multiple models, graph analysis, and structured output all working together. The scoreboard provides a clear, presentable summary of what the system produces, while the centrality analysis shows deeper network insights.

**Output:** Saves `pipeline/7_advanced_graph.png`.

---

## Quick Reference

| Command | What You Get |
|---|---|
| `python run_recommend.py` → Option 1 | Full hybrid recommendations for one user |
| `python run_recommend.py` → Option 2 | Global trending events (no user needed) |
| `python run_recommend.py` → Option 3 | Graph-only collaborative filtering for one user |
| `python run_recommend.py` → Option 4 | Hybrid + custom interest/city explanations |
| `python run_pipeline.py` → Sections 1–7 | Complete system walkthrough with visuals and metrics |
