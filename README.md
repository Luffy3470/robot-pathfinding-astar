# Robot & Multi-Robot Path-Finding with A* (CSMI17 – AI Assignment)

Implementation of grid-based robot path-finding using A* search
(Question 1: single robot, 3+ heuristics compared) and multi-robot
path-finding (Question 2: naive independent A* vs. a proposed
Cooperative Space-Time A* method).

## Structure

```
app.py                           # interactive Streamlit app (run this for a live demo)
src/
  grid.py                 # random solvable grid / instance generation
  heuristics.py            # manhattan, euclidean, chebyshev, dijkstra-baseline
  astar.py                  # single-robot A* (Q1)
  experiment_q1.py          # Q1 experiment runner -> results/q1_results.csv
  plot_q1.py                 # Q1 plots + summary table
  multi_robot.py             # plain multi-robot A* + Cooperative Space-Time A* (Q2)
  experiment_q2.py           # Q2 experiment runner -> results/q2_results.csv
  plot_q2.py                  # Q2 plots + summary table
  visualize_examples.py       # single-instance "screenshots" used in the report
results/                       # generated CSVs, summary tables and PNG plots
report_assets/                  # (optional) extra images used only in the report
Report.docx                      # full assignment report
Presentation.pptx                 # slide deck
requirements.txt
```

## Running the interactive demo (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser-based app with three pages (pick from the sidebar):

- **Q1 — Single-Robot A\*** — pick a grid size, obstacle density, seed, and
  which heuristics to compare; generates a random solvable grid and solves
  it live, showing each heuristic's path and a nodes-expanded comparison.
- **Q2 — Multi-Robot Path Finding** — pick grid size, density, number of
  robots, and a seed; plans the same scenario with both the plain A*
  baseline and the proposed Cooperative Space-Time A*, side by side, with
  collision counts and cost metrics.
- **Aggregate Study Results** — the pre-computed charts and summary tables
  from the full 540-run (Q1) / 384-run (Q2) experiment sweep used in the
  report.

## Running the batch experiments (no UI)

```bash
pip install -r requirements.txt

# Question 1
python src/experiment_q1.py     # generates results/q1_results.csv
python src/plot_q1.py           # generates plots + results/q1_summary.csv

# Question 2
python src/experiment_q2.py     # generates results/q2_results.csv
python src/plot_q2.py           # generates plots + results/q2_summary.csv

# Example screenshots used in the report
python src/visualize_examples.py
```

## Question 1 — Single robot A*

Three admissible heuristics are implemented and compared (plus a
Dijkstra/`h=0` baseline for reference):

- **Manhattan distance** — `|dr| + |dc|`
- **Euclidean distance** — straight-line distance
- **Chebyshev distance** — `max(|dr|, |dc|)`

Metrics compared: nodes expanded, nodes generated, runtime, and path
optimality (all three heuristics are admissible, so all return
shortest paths — they only differ in search effort/time).

## Question 2 — Multi-robot path finding

- **Baseline ("plain A*")**: every robot plans its own shortest path
  independently with single-robot A*, with no knowledge of the other
  robots. When executed together this frequently produces **vertex
  collisions** (two robots in the same cell at the same time) and
  **edge/swap collisions** (two robots crossing paths and swapping
  cells between consecutive timesteps).

- **Proposed method — Cooperative Space-Time A***: robots are planned
  one at a time in priority order. Each robot searches over
  `(row, col, time)` states instead of just `(row, col)`, and treats
  the already-committed paths of higher-priority robots (read from a
  shared reservation table) as **dynamic obstacles**. This lets a
  robot wait for another to clear a cell, and guarantees the final
  joint plan is fully collision-free (both vertex and edge/swap
  collisions), at the cost of no longer being minimal-total-cost for
  the whole team (it is optimal only conditioned on the priority
  order).

See `Report.docx` for the full write-up: problem definition,
assumptions, algorithm description, experimental setup with
screenshots, and performance comparison.
