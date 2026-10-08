import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from grid import generate_instance, generate_multi_agent_instance
from astar import astar
from heuristics import manhattan
from multi_robot import plain_astar_multi, cooperative_astar_multi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def draw_grid(ax, grid):
    rows, cols = grid.shape
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------
# Q1 screenshot: single robot grid + A* path with Manhattan heuristic
# ---------------------------------------------------------------
grid, start, goal = generate_instance(20, 20, 0.22, seed=42)
result = astar(grid, start, goal, manhattan)
path = result["path"]

fig, ax = plt.subplots(figsize=(6, 6))
draw_grid(ax, grid)
ys = [p[0] for p in path]
xs = [p[1] for p in path]
ax.plot(xs, ys, color="#2b6cb0", linewidth=2.5, marker="o", markersize=3, label="A* path")
ax.scatter([start[1]], [start[0]], color="green", s=140, marker="s", zorder=5, label="Start")
ax.scatter([goal[1]], [goal[0]], color="red", s=140, marker="*", zorder=5, label="Goal")
ax.set_title(f"Q1: Single-Robot A* (Manhattan heuristic)\npath length={result['path_length']}, "
             f"nodes expanded={result['nodes_expanded']}")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "screenshot_q1_grid_path.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved Q1 screenshot")

# ---------------------------------------------------------------
# Q2 screenshots: multi-robot scenario - plain A* (with collisions)
# vs cooperative A* (collision-free) on the SAME instance
# ---------------------------------------------------------------
mgrid, starts, goals = generate_multi_agent_instance(16, 16, 8, 0.15, seed=7)
plain = plain_astar_multi(mgrid, starts, goals)
coop = cooperative_astar_multi(mgrid, starts, goals)

cmap = plt.get_cmap("tab10")


def draw_multi(ax, grid, paths, starts, goals, title, collisions=None):
    draw_grid(ax, grid)
    for i, p in enumerate(paths):
        color = cmap(i % 10)
        ys = [c[0] for c in p]
        xs = [c[1] for c in p]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.85)
        ax.scatter([starts[i][1]], [starts[i][0]], color=color, s=90, marker="s", edgecolor="black", zorder=5)
        ax.scatter([goals[i][1]], [goals[i][0]], color=color, s=140, marker="*", edgecolor="black", zorder=5)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
draw_multi(axes[0], mgrid, plain["paths"], starts, goals,
           f"Q2: Plain A* (independent)\ncollisions: {plain['vertex_collisions']} vertex, "
           f"{plain['edge_collisions']} edge")
draw_multi(axes[1], mgrid, coop["paths"], starts, goals,
           f"Q2: Cooperative Space-Time A* (proposed)\ncollisions: {coop['vertex_collisions']} vertex, "
           f"{coop['edge_collisions']} edge")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "screenshot_q2_multi_robot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved Q2 screenshot")
print("plain collisions:", plain["vertex_collisions"], plain["edge_collisions"])
print("coop collisions:", coop["vertex_collisions"], coop["edge_collisions"])
