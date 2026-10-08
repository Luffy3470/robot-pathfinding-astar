"""
experiment_q2.py
Runs the multi-robot experiment (Question 2): compares the naive
"plain A*" baseline against the proposed Cooperative Space-Time A*
across multiple grid sizes, obstacle densities, and agent counts.
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(__file__))
from grid import generate_multi_agent_instance
from multi_robot import plain_astar_multi, cooperative_astar_multi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GRID_SIZES = [20, 30]
DENSITIES = [0.1, 0.2]
AGENT_COUNTS = [2, 4, 6, 8, 12, 16]
TRIALS_PER_CONFIG = 8


def run():
    rows_out = []
    trial_id = 0
    for size in GRID_SIZES:
        for density in DENSITIES:
            for n_agents in AGENT_COUNTS:
                for t in range(TRIALS_PER_CONFIG):
                    seed = trial_id * 131 + 7
                    trial_id += 1
                    try:
                        grid, starts, goals = generate_multi_agent_instance(
                            size, size, n_agents, density, seed=seed)
                    except RuntimeError:
                        continue

                    plain = plain_astar_multi(grid, starts, goals)
                    coop = cooperative_astar_multi(grid, starts, goals)

                    for method_result in (plain, coop):
                        rows_out.append({
                            "trial_id": trial_id,
                            "grid_size": size,
                            "density": density,
                            "n_agents": n_agents,
                            "method": method_result["method"],
                            "individual_search_success": method_result["individual_search_success"],
                            "collision_free": method_result["collision_free"],
                            "vertex_collisions": method_result["vertex_collisions"],
                            "edge_collisions": method_result["edge_collisions"],
                            "makespan": method_result["makespan"],
                            "sum_of_costs": method_result["sum_of_costs"],
                            "nodes_expanded": method_result["nodes_expanded"],
                            "time_sec": method_result["time_sec"],
                        })
                print(f"size={size} density={density} n_agents={n_agents}: done")

    out_path = os.path.join(RESULTS_DIR, "q2_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {len(rows_out)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    run()
