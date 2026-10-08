"""
astar.py
Generic A* search on a 4-connected 2D grid (single robot, Question 1).

Returns a dict of performance metrics so different heuristics can be
compared fairly:
    path            list of (r,c) cells from start to goal (None if no path)
    path_length     number of steps in the path (edges), -1 if unreachable
    cost            total path cost (== path_length, unit edge cost)
    nodes_expanded  number of nodes popped off the open list (popped &
                    closed) -- a standard proxy for search effort
    nodes_generated number of nodes pushed onto the open list (incl. dups)
    time_sec        wall-clock time for the search
    success         bool
"""
import heapq
import time


MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid, start, goal, heuristic):
    """
    grid: 2D numpy array, 0 free / 1 obstacle
    start, goal: (row, col) tuples
    heuristic: function (a, b) -> float, admissible estimate of cost(a,b)
    """
    rows, cols = grid.shape
    t0 = time.perf_counter()

    open_heap = []
    counter = 0  # tie-breaker to keep heap comparisons well-defined
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    nodes_expanded = 0
    nodes_generated = 1

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            t1 = time.perf_counter()
            return {
                "path": path,
                "path_length": len(path) - 1,
                "cost": g_score[current],
                "nodes_expanded": nodes_expanded,
                "nodes_generated": nodes_generated,
                "time_sec": t1 - t0,
                "success": True,
            }

        r, c = current
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            neighbor = (nr, nc)
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))
                nodes_generated += 1

    t1 = time.perf_counter()
    return {
        "path": None,
        "path_length": -1,
        "cost": float("inf"),
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "time_sec": t1 - t0,
        "success": False,
    }
