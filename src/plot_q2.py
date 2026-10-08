import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RESULTS_DIR, "q2_results.csv"))

COLORS = {"plain_astar": "#c05621", "cooperative_astar": "#2f855a"}
LABELS = {"plain_astar": "Plain A* (independent)", "cooperative_astar": "Cooperative Space-Time A* (proposed)"}

# ---- 1. Collision-free rate vs number of agents ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = df[df.method == method].groupby("n_agents")["collision_free"].mean() * 100
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Number of robots")
ax.set_ylabel("Collision-free trials (%)")
ax.set_title("Collision-Free Rate vs Team Size")
ax.set_ylim(-5, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_collision_free_rate.png"), dpi=150)
plt.close()

# ---- 2. Total collisions (vertex + edge) vs number of agents ----
df["total_collisions"] = df["vertex_collisions"] + df["edge_collisions"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = df[df.method == method].groupby("n_agents")["total_collisions"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Number of robots")
ax.set_ylabel("Avg. collisions per trial")
ax.set_title("Average Collisions vs Team Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_collisions_vs_agents.png"), dpi=150)
plt.close()

# ---- 3. Sum-of-costs vs number of agents (efficiency cost of coordination) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = df[df.method == method].groupby("n_agents")["sum_of_costs"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Number of robots")
ax.set_ylabel("Avg. sum of costs (total path length)")
ax.set_title("Solution Cost vs Team Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_sum_of_costs.png"), dpi=150)
plt.close()

# ---- 4. Makespan vs number of agents ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = df[df.method == method].groupby("n_agents")["makespan"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Number of robots")
ax.set_ylabel("Avg. makespan (timesteps)")
ax.set_title("Makespan vs Team Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_makespan.png"), dpi=150)
plt.close()

# ---- 5. Computation time vs number of agents ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = df[df.method == method].groupby("n_agents")["time_sec"].mean() * 1000
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Number of robots")
ax.set_ylabel("Avg. planning time (ms)")
ax.set_title("Planning Time vs Team Size")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_time_vs_agents.png"), dpi=150)
plt.close()

# ---- 6. Collision-free rate vs obstacle density (n_agents fixed at max) ----
sub_df = df[df.n_agents == df.n_agents.max()]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for method in ["plain_astar", "cooperative_astar"]:
    sub = sub_df[sub_df.method == method].groupby("density")["collision_free"].mean() * 100
    ax.plot(sub.index, sub.values, marker="o", label=LABELS[method], color=COLORS[method])
ax.set_xlabel("Obstacle density")
ax.set_ylabel("Collision-free trials (%)")
ax.set_title(f"Collision-Free Rate vs Density ({int(df.n_agents.max())} robots)")
ax.set_ylim(-5, 105)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "q2_collision_free_vs_density.png"), dpi=150)
plt.close()

# ---- Summary tables ----
summary = df.groupby(["method", "n_agents"]).agg(
    collision_free_rate=("collision_free", "mean"),
    avg_collisions=("total_collisions", "mean"),
    avg_makespan=("makespan", "mean"),
    avg_sum_of_costs=("sum_of_costs", "mean"),
    avg_time_ms=("time_sec", lambda x: x.mean() * 1000),
).reset_index()
summary.to_csv(os.path.join(RESULTS_DIR, "q2_summary.csv"), index=False)
print(summary.to_string(index=False))
