"""
app.py
Interactive Streamlit front-end for the CSMI17 AI assignment:
  - Q1: single-robot A* search with a choice of heuristic, on a
    randomly generated (or reproducible-via-seed) grid.
  - Q2: multi-robot path-finding, comparing the naive "plain A*"
    baseline against the proposed Cooperative Space-Time A* on the
    same randomly generated multi-agent instance.
  - Study: the pre-computed aggregate results (540 Q1 runs / 384 Q2
    runs) from the full experiment sweep, as static charts.

Run with:
    streamlit run app.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from grid import generate_instance, generate_multi_agent_instance
from astar import astar
from heuristics import HEURISTICS
from multi_robot import plain_astar_multi, cooperative_astar_multi

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CMAP = plt.get_cmap("tab10")

st.set_page_config(page_title="Robot Path-Finding with A*", page_icon="\U0001F916", layout="wide")


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------
def draw_grid(ax, grid):
    rows, cols = grid.shape
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_single_path(grid, start, goal, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_grid(ax, grid)
    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="#2b6cb0", linewidth=2.5, marker="o", markersize=3)
    ax.scatter([start[1]], [start[0]], color="green", s=140, marker="s", zorder=5, label="Start")
    ax.scatter([goal[1]], [goal[0]], color="red", s=140, marker="*", zorder=5, label="Goal")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def draw_multi(grid, paths, starts, goals, title):
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    draw_grid(ax, grid)
    for i, p in enumerate(paths):
        color = CMAP(i % 10)
        ys = [c[0] for c in p]
        xs = [c[1] for c in p]
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.9)
        ax.scatter([starts[i][1]], [starts[i][0]], color=color, s=80, marker="s", edgecolor="black", zorder=5)
        ax.scatter([goals[i][1]], [goals[i][0]], color=color, s=130, marker="*", edgecolor="black", zorder=5)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
st.sidebar.title("\U0001F916 Robot Path-Finding")
st.sidebar.caption("CSMI17 \u2014 Artificial Intelligence, Assignment 5")
page = st.sidebar.radio("Go to", [
    "Q1 \u2014 Single-Robot A*",
    "Q2 \u2014 Multi-Robot Path Finding",
    "Aggregate Study Results",
])

# =========================================================================
# PAGE 1 — Q1
# =========================================================================
if page == "Q1 \u2014 Single-Robot A*":
    st.title("Q1: Single-Robot A* Search")
    st.write(
        "Generate a random grid and solve it with A*, comparing heuristics. "
        "All heuristics shown are admissible, so they all return an optimal "
        "path \u2014 they differ only in how many nodes they need to expand."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        size = st.slider("Grid size (N x N)", 10, 60, 25)
    with c2:
        density = st.slider("Obstacle density", 0.0, 0.45, 0.2, step=0.05)
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    with c4:
        chosen = st.multiselect(
            "Heuristics to compare",
            list(HEURISTICS.keys()),
            default=["manhattan", "euclidean", "chebyshev", "dijkstra_baseline"],
        )

    if st.button("Generate grid & solve", type="primary"):
        with st.spinner("Generating a solvable instance..."):
            grid, start, goal = generate_instance(size, size, density, seed=int(seed))
        st.session_state["q1_grid"] = (grid, start, goal)

    if "q1_grid" in st.session_state and chosen:
        grid, start, goal = st.session_state["q1_grid"]
        rows = []
        cols = st.columns(len(chosen))
        for col, hname in zip(cols, chosen):
            hfunc = HEURISTICS[hname]
            result = astar(grid, start, goal, hfunc)
            rows.append({
                "Heuristic": hname,
                "Path length": result["path_length"],
                "Nodes expanded": result["nodes_expanded"],
                "Nodes generated": result["nodes_generated"],
                "Time (ms)": round(result["time_sec"] * 1000, 3),
            })
            with col:
                fig = draw_single_path(
                    grid, start, goal, result["path"],
                    f"{hname}\nnodes={result['nodes_expanded']}, len={result['path_length']}",
                )
                st.pyplot(fig)
                plt.close(fig)

        st.subheader("Metrics")
        df = pd.DataFrame(rows).sort_values("Nodes expanded")
        st.dataframe(df, width='stretch', hide_index=True)
        st.bar_chart(df.set_index("Heuristic")["Nodes expanded"])
    elif "q1_grid" not in st.session_state:
        st.info("Set your parameters above and click **Generate grid & solve**.")


# =========================================================================
# PAGE 2 — Q2
# =========================================================================
elif page == "Q2 \u2014 Multi-Robot Path Finding":
    st.title("Q2: Multi-Robot Path Finding")
    st.write(
        "Compare the naive **plain A\\*** baseline (each robot plans alone, "
        "unaware of the others) against the proposed **Cooperative Space-Time "
        "A\\*** (robots planned sequentially over a reservation table) on the "
        "same random multi-robot instance."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        size = st.slider("Grid size (N x N)", 10, 40, 18, key="q2_size")
    with c2:
        density = st.slider("Obstacle density", 0.0, 0.35, 0.15, step=0.05, key="q2_density")
    with c3:
        n_agents = st.slider("Number of robots", 2, 20, 8, key="q2_agents")
    with c4:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=7, step=1, key="q2_seed")

    if st.button("Generate scenario & plan", type="primary"):
        with st.spinner("Generating a solvable multi-robot instance..."):
            try:
                grid, starts, goals = generate_multi_agent_instance(
                    size, size, n_agents, density, seed=int(seed))
                st.session_state["q2_scenario"] = (grid, starts, goals)
            except RuntimeError as e:
                st.error(f"Could not generate a solvable instance: {e}. Try a lower density or fewer robots.")

    if "q2_scenario" in st.session_state:
        grid, starts, goals = st.session_state["q2_scenario"]
        t0 = time.perf_counter()
        plain = plain_astar_multi(grid, starts, goals)
        coop = cooperative_astar_multi(grid, starts, goals)

        colA, colB = st.columns(2)
        with colA:
            fig = draw_multi(
                grid, plain["paths"], starts, goals,
                f"Plain A* (independent)\n{plain['vertex_collisions']} vertex + "
                f"{plain['edge_collisions']} edge collisions",
            )
            st.pyplot(fig)
            plt.close(fig)
        with colB:
            fig = draw_multi(
                grid, coop["paths"], starts, goals,
                f"Cooperative Space-Time A* (proposed)\n{coop['vertex_collisions']} vertex + "
                f"{coop['edge_collisions']} edge collisions",
            )
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Metrics")
        metrics = pd.DataFrame([
            {
                "Method": "Plain A* (independent)",
                "Collision-free?": "Yes" if plain["collision_free"] else "No",
                "Vertex collisions": plain["vertex_collisions"],
                "Edge collisions": plain["edge_collisions"],
                "Makespan": plain["makespan"],
                "Sum of costs": plain["sum_of_costs"],
                "Planning time (ms)": round(plain["time_sec"] * 1000, 2),
            },
            {
                "Method": "Cooperative Space-Time A* (proposed)",
                "Collision-free?": "Yes" if coop["collision_free"] else "No",
                "Vertex collisions": coop["vertex_collisions"],
                "Edge collisions": coop["edge_collisions"],
                "Makespan": coop["makespan"],
                "Sum of costs": coop["sum_of_costs"],
                "Planning time (ms)": round(coop["time_sec"] * 1000, 2),
            },
        ])
        st.dataframe(metrics, width='stretch', hide_index=True)

        if plain["collision_free"] and not coop["collision_free"]:
            st.warning("This particular random instance happened to be collision-free for both methods.")
        elif not plain["collision_free"] and coop["collision_free"]:
            st.success(
                "The plain baseline produced collisions on this instance; the proposed "
                "Cooperative Space-Time A* resolved them, confirming the pattern seen "
                "across the full 192-instance study."
            )
    else:
        st.info("Set your parameters above and click **Generate scenario & plan**.")


# =========================================================================
# PAGE 3 — Aggregate study results (pre-computed, from the full report)
# =========================================================================
else:
    st.title("Aggregate Study Results")
    st.write(
        "These charts summarize the full experiment sweep used in the report: "
        "**540** single-robot runs (Q1) and **384** multi-robot planning runs (Q2)."
    )

    tab1, tab2 = st.tabs(["Q1 \u2014 Heuristic comparison", "Q2 \u2014 Plain vs. Cooperative A*"])

    with tab1:
        q1_summary_path = os.path.join(RESULTS_DIR, "q1_summary.csv")
        if os.path.exists(q1_summary_path):
            st.dataframe(pd.read_csv(q1_summary_path), width='stretch', hide_index=True)
        cols = st.columns(2)
        imgs = ["q1_nodes_expanded_avg.png", "q1_nodes_vs_size.png",
                "q1_nodes_vs_density.png", "q1_time_vs_size.png"]
        for i, img in enumerate(imgs):
            p = os.path.join(RESULTS_DIR, img)
            if os.path.exists(p):
                cols[i % 2].image(p, width='stretch')

    with tab2:
        q2_summary_path = os.path.join(RESULTS_DIR, "q2_summary.csv")
        if os.path.exists(q2_summary_path):
            st.dataframe(pd.read_csv(q2_summary_path), width='stretch', hide_index=True)
        cols = st.columns(2)
        imgs = ["q2_collision_free_rate.png", "q2_sum_of_costs.png",
                "q2_collisions_vs_agents.png", "q2_makespan.png",
                "q2_time_vs_agents.png", "q2_collision_free_vs_density.png"]
        for i, img in enumerate(imgs):
            p = os.path.join(RESULTS_DIR, img)
            if os.path.exists(p):
                cols[i % 2].image(p, width='stretch')
