import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RESULTS_DIR, "q1_results.csv"))

HEUR_ORDER = ["dijkstra_baseline", "manhattan", "euclidean", "chebyshev"]
COLORS = {"dijkstra_baseline": "#888888", "manhattan": "#2b6cb0",
          "euclidean": "#c05621", "chebyshev": "#2f855a"}

# ---- 1. Nodes expanded by heuristic (bar, averaged over all configs) ----
agg = df.groupby("heuristic")["nodes_expanded"].mean().reindex(HEUR_ORDER)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(agg.index, agg.values, color=[COLORS[h] for h in agg.index])
ax.set_ylabel("Avg. nodes expanded")
ax.set_title("Average Search Effort by Heuristic (all grid sizes/densities)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q1_nodes_expanded_avg.png"), dpi=150)
plt.close()

# ---- 2. Nodes expanded vs grid size, per heuristic ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for h in HEUR_ORDER:
    sub = df[df.heuristic == h].groupby("grid_size")["nodes_expanded"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=h, color=COLORS[h])
ax.set_xlabel("Grid size (N x N)")
ax.set_ylabel("Avg. nodes expanded")
ax.set_title("Search Effort vs Grid Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q1_nodes_vs_size.png"), dpi=150)
plt.close()

# ---- 3. Runtime vs grid size, per heuristic ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for h in HEUR_ORDER:
    sub = df[df.heuristic == h].groupby("grid_size")["time_sec"].mean() * 1000
    ax.plot(sub.index, sub.values, marker="o", label=h, color=COLORS[h])
ax.set_xlabel("Grid size (N x N)")
ax.set_ylabel("Avg. runtime (ms)")
ax.set_title("Runtime vs Grid Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q1_time_vs_size.png"), dpi=150)
plt.close()

# ---- 4. Nodes expanded vs obstacle density ----
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for h in HEUR_ORDER:
    sub = df[df.heuristic == h].groupby("density")["nodes_expanded"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=h, color=COLORS[h])
ax.set_xlabel("Obstacle density")
ax.set_ylabel("Avg. nodes expanded")
ax.set_title("Search Effort vs Obstacle Density")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q1_nodes_vs_density.png"), dpi=150)
plt.close()

# ---- 5. Path length sanity check (should be ~equal across heuristics -> all optimal) ----
piv = df.pivot_table(index="trial_id", columns="heuristic", values="path_length")
optimal_match = (piv[HEUR_ORDER].eq(piv["dijkstra_baseline"], axis=0)).mean()
print("Fraction of trials where heuristic path_length matches optimal (Dijkstra) baseline:")
print(optimal_match)

# ---- 6. Summary table ----
summary = df.groupby("heuristic").agg(
    avg_nodes_expanded=("nodes_expanded", "mean"),
    avg_nodes_generated=("nodes_generated", "mean"),
    avg_time_ms=("time_sec", lambda x: x.mean() * 1000),
    avg_path_length=("path_length", "mean"),
    success_rate=("success", "mean"),
).reindex(HEUR_ORDER)
summary.to_csv(os.path.join(RESULTS_DIR, "q1_summary.csv"))
print(summary)
