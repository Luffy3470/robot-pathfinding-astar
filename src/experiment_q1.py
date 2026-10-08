"""
experiment_q1.py
Runs the single-robot A* experiment (Question 1):
  - generates many random solvable grids (varying size & obstacle density)
  - solves each with every heuristic in heuristics.HEURISTICS
  - records nodes_expanded, nodes_generated, time_sec, path_length
  - saves raw results to results/q1_results.csv
  - saves summary plots to results/
"""
import os
import sys
import csv
import statistics as stats

sys.path.insert(0, os.path.dirname(__file__))
from grid import generate_instance
from astar import astar
from heuristics import HEURISTICS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [15, 25, 40]          # rows == cols
DENSITIES = [0.1, 0.2, 0.3]
TRIALS_PER_CONFIG = 15             # random instances per (size, density) combo


def run():
    rows_out = []
    trial_id = 0
    for size in GRID_SIZES:
        for density in DENSITIES:
            for t in range(TRIALS_PER_CONFIG):
                seed = trial_id * 97 + 1
                trial_id += 1
                grid, start, goal = generate_instance(size, size, density, seed=seed)
                for hname, hfunc in HEURISTICS.items():
                    result = astar(grid, start, goal, hfunc)
                    rows_out.append({
                        "trial_id": trial_id,
                        "grid_size": size,
                        "density": density,
                        "heuristic": hname,
                        "success": result["success"],
                        "path_length": result["path_length"],
                        "nodes_expanded": result["nodes_expanded"],
                        "nodes_generated": result["nodes_generated"],
                        "time_sec": result["time_sec"],
                    })
                if t == 0:
                    print(f"size={size} density={density}: sample trial done")

    out_path = os.path.join(RESULTS_DIR, "q1_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    run()
