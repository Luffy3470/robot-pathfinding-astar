"""
grid.py
Utilities to generate random grid-world environments for the
robot path-finding assignment (Q1) and multi-robot path-finding (Q2).

A grid is represented as a 2D numpy array of shape (rows, cols):
    0 -> free cell
    1 -> obstacle

Coordinates are (row, col), row increases downward.
"""
import random
import numpy as np


def generate_grid(rows, cols, obstacle_density=0.2, seed=None):
    """Generate a random grid with the given obstacle density (0..1)."""
    rng = random.Random(seed)
    grid = np.zeros((rows, cols), dtype=np.int8)
    n_obstacles = int(obstacle_density * rows * cols)
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    for (r, c) in cells[:n_obstacles]:
        grid[r, c] = 1
    return grid


def random_free_cell(grid, rng, forbidden=frozenset()):
    """Pick a random free cell not in `forbidden`."""
    rows, cols = grid.shape
    free = [(r, c) for r in range(rows) for c in range(cols)
            if grid[r, c] == 0 and (r, c) not in forbidden]
    if not free:
        return None
    return rng.choice(free)


def is_reachable(grid, start, goal):
    """BFS reachability check (4-connected) used to guarantee solvable instances."""
    from collections import deque
    rows, cols = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return False
    visited = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return False


def generate_instance(rows, cols, obstacle_density=0.2, seed=None, max_tries=200):
    """Generate a (grid, start, goal) instance guaranteed to be solvable."""
    rng = random.Random(seed)
    for attempt in range(max_tries):
        grid = generate_grid(rows, cols, obstacle_density, seed=(seed if seed is None else seed + attempt))
        start = random_free_cell(grid, rng)
        goal = random_free_cell(grid, rng, forbidden={start} if start else frozenset())
        if start is None or goal is None:
            continue
        if is_reachable(grid, start, goal):
            return grid, start, goal
    raise RuntimeError("Could not generate a solvable instance; lower obstacle_density.")


def generate_multi_agent_instance(rows, cols, n_agents, obstacle_density=0.2, seed=None, max_tries=300):
    """
    Generate a grid with n_agents distinct start cells and n_agents distinct
    goal cells (starts and goals may overlap with each other's goals/starts,
    but every (start_i, goal_i) pair must be individually reachable).
    """
    rng = random.Random(seed)
    for attempt in range(max_tries):
        grid = generate_grid(rows, cols, obstacle_density, seed=(seed if seed is None else seed + attempt))
        used = set()
        starts, goals = [], []
        ok = True
        for i in range(n_agents):
            s = random_free_cell(grid, rng, forbidden=used)
            if s is None:
                ok = False
                break
            used.add(s)
            g = random_free_cell(grid, rng, forbidden=used)
            if g is None:
                ok = False
                break
            used.add(g)
            if not is_reachable(grid, s, g):
                ok = False
                break
            starts.append(s)
            goals.append(g)
        if ok:
            return grid, starts, goals
    raise RuntimeError("Could not generate a solvable multi-agent instance; lower density or agents.")
