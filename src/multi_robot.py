"""
multi_robot.py
Question 2: Multi-Robot Path Finding.

Two methods are implemented:

1. plain_astar_multi()
   Each robot independently runs the single-robot A* from Q1 (Manhattan
   heuristic, the best performer from Q1) completely ignoring the other
   robots. The resulting paths are then "executed" together and checked
   for collisions. This is the naive baseline requested by the assignment
   ("try the A* algorithm for this setting").

2. cooperative_astar_multi()  -- the proposed method
   A well-known, simple-but-effective technique from the multi-agent
   path-finding (MAPF) literature: Cooperative A* / Space-Time A*
   with a reservation table (Silver, 2005 - "Cooperative Pathfinding").

   - Robots are planned ONE AT A TIME, in a priority order.
   - Each robot's search state is (row, col, t) i.e. space-time, not just
     space. This lets a robot wait for another robot to clear a cell.
   - After a robot's path is planned, its (cell, time) occupancy and the
     edges it traverses are written into a shared reservation table.
   - Every subsequent robot's A* treats reserved (cell, time) pairs, and
     reserved edge-swaps, as forbidden -- i.e. as *dynamic* obstacles.
   - This guarantees the final joint plan is collision-free (no vertex
     collisions, no edge/swap collisions) whenever every individual
     space-time search succeeds, at the cost of no longer being optimal
     for the team as a whole (only optimal *given* the priority order).

Both methods return a dict of metrics so they can be compared head-to-head
by experiment_q2.py.
"""
import heapq
import itertools
import time

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # N,S,W,E, WAIT


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------------------------------------------------
# Collision checking for a set of already-computed paths (used to grade
# the "plain A*" baseline, which plans with no knowledge of other robots)
# ----------------------------------------------------------------------
def count_collisions(paths):
    """
    paths: list of list[(r,c)], one per robot (shorter paths are assumed
    to stay at their goal cell after finishing, i.e. robots don't vanish).
    Returns (n_vertex_collisions, n_edge_collisions) counted over time.
    """
    if not paths:
        return 0, 0
    T = max(len(p) for p in paths)

    def pos_at(path, t):
        if t < len(path):
            return path[t]
        return path[-1]  # robot waits at goal after arrival

    vertex_collisions = 0
    edge_collisions = 0
    for t in range(T):
        occ = {}
        for i, p in enumerate(paths):
            cell = pos_at(p, t)
            occ.setdefault(cell, []).append(i)
        for cell, robots in occ.items():
            if len(robots) > 1:
                vertex_collisions += len(robots) - 1
        if t > 0:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    a_prev, a_cur = pos_at(paths[i], t - 1), pos_at(paths[i], t)
                    b_prev, b_cur = pos_at(paths[j], t - 1), pos_at(paths[j], t)
                    if a_prev == b_cur and a_cur == b_prev and a_prev != a_cur:
                        edge_collisions += 1
    return vertex_collisions, edge_collisions


# ----------------------------------------------------------------------
# 1) Plain / naive multi-robot A*  (baseline)
# ----------------------------------------------------------------------
def single_astar(grid, start, goal, heuristic=manhattan):
    rows, cols = grid.shape
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, nodes_expanded
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                neighbor = (nr, nc)
                tentative = g_score[current] + 1
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    counter += 1
                    heapq.heappush(open_heap, (tentative + heuristic(neighbor, goal), counter, neighbor))
    return None, nodes_expanded


def plain_astar_multi(grid, starts, goals):
    t0 = time.perf_counter()
    paths = []
    total_nodes = 0
    all_success = True
    for s, g in zip(starts, goals):
        path, nodes = single_astar(grid, s, g)
        total_nodes += nodes
        if path is None:
            all_success = False
            path = [s]
        paths.append(path)
    t1 = time.perf_counter()
    vcol, ecol = count_collisions(paths)
    makespan = max(len(p) - 1 for p in paths) if paths else 0
    sum_of_costs = sum(len(p) - 1 for p in paths)
    return {
        "method": "plain_astar",
        "paths": paths,
        "individual_search_success": all_success,
        "collision_free": (vcol == 0 and ecol == 0),
        "vertex_collisions": vcol,
        "edge_collisions": ecol,
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "nodes_expanded": total_nodes,
        "time_sec": t1 - t0,
    }


# ----------------------------------------------------------------------
# 2) Proposed method: Cooperative Space-Time A* with reservation table
# ----------------------------------------------------------------------
def space_time_astar(grid, start, goal, reserved_vertices, reserved_edges,
                      max_time, heuristic=manhattan):
    """
    Search over states (row, col, t). An agent may move to a 4-neighbour
    or wait in place; cost is 1 per timestep (including waiting), which
    encourages progress while still allowing waits when required.

    reserved_vertices: set of (r, c, t) already occupied by earlier robots
                        (a robot also reserves its goal cell for all
                        t >= arrival time, so later robots don't plan
                        through a parked robot).
    reserved_edges:    set of (r, c, r2, c2, t) meaning "moving from
                        (r,c) to (r2,c2) between t-1 and t is forbidden"
                        (prevents head-on swap collisions).
    """
    rows, cols = grid.shape
    open_heap = []
    counter = 0
    start_state = (start[0], start[1], 0)
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start_state))
    came_from = {}
    g_score = {start_state: 0}
    closed = set()
    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_expanded += 1
        r, c, t = current

        # A robot that "arrives" at its goal is assumed to PARK there for
        # the rest of the horizon (it does not vanish). So reaching the
        # goal cell is only a valid solution if that cell is free of
        # reservations for every future timestep too -- otherwise the
        # robot would eventually collide with an already-planned robot
        # that later passes through/park there. If it's not safe yet,
        # we keep searching (the robot must arrive later, e.g. by
        # taking a longer route or waiting elsewhere first).
        if (r, c) == goal and all(
                (r, c, t2) not in reserved_vertices for t2 in range(t + 1, max_time + 1)):
            path = [(r, c)]
            cur = current
            while cur in came_from:
                cur = came_from[cur]
                path.append((cur[0], cur[1]))
            path.reverse()
            return path, nodes_expanded

        if t >= max_time:
            continue

        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] == 1:
                continue
            nt = t + 1
            if (nr, nc, nt) in reserved_vertices:
                continue
            if (r, c, nr, nc, nt) in reserved_edges:
                continue
            neighbor = (nr, nc, nt)
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                f = tentative + heuristic((nr, nc), goal)
                heapq.heappush(open_heap, (f, counter, neighbor))
    return None, nodes_expanded


def cooperative_astar_multi(grid, starts, goals, priority_order=None, horizon_slack=4):
    """
    Plans robots sequentially (Cooperative A*). Returns the same metric
    dict shape as plain_astar_multi() for direct comparison.
    """
    t0 = time.perf_counter()
    n = len(starts)
    order = priority_order if priority_order is not None else list(range(n))

    reserved_vertices = set()
    reserved_edges = set()
    paths = [None] * n
    total_nodes = 0
    all_success = True

    # Rough horizon: give each robot generous slack over its Manhattan
    # distance to account for detours around obstacles/other robots.
    rows, cols = grid.shape
    base_horizon = rows + cols
    max_time = base_horizon * horizon_slack

    # IMPORTANT: every robot physically occupies its own start cell until
    # it is actually given a plan, even though it hasn't been "planned"
    # yet in this priority ordering. We model every not-yet-planned
    # robot as parked (a temporary obstacle) at its start cell for the
    # whole horizon; this is removed for a robot right before it is
    # planned, and replaced with its real space-time path afterwards.
    # This prevents earlier-planned (higher-priority) robots from
    # routing straight through a lower-priority robot's starting cell.
    pending_start_blocks = {}
    for i, s in enumerate(starts):
        block = {(s[0], s[1], t) for t in range(max_time + 1)}
        pending_start_blocks[i] = block
        reserved_vertices |= block

    for idx in order:
        s, g = starts[idx], goals[idx]
        # This robot is about to be planned: stop treating its own start
        # cell as a blanket obstacle.
        reserved_vertices -= pending_start_blocks[idx]

        path, nodes = space_time_astar(grid, s, g, reserved_vertices, reserved_edges, max_time)
        total_nodes += nodes
        if path is None:
            all_success = False
            path = [s]  # robot stays put; leave a placeholder
        paths[idx] = path

        # Reserve this robot's space-time path so later robots avoid it.
        for t, cell in enumerate(path):
            reserved_vertices.add((cell[0], cell[1], t))
        # After arrival, keep the goal cell reserved for the rest of the
        # horizon (the robot "parks" there).
        for t in range(len(path), max_time + 1):
            reserved_vertices.add((path[-1][0], path[-1][1], t))
        # Reserve edges (both directions) to forbid swap collisions.
        for t in range(1, len(path)):
            a, b = path[t - 1], path[t]
            reserved_edges.add((a[0], a[1], b[0], b[1], t))
            reserved_edges.add((b[0], b[1], a[0], a[1], t))

    t1 = time.perf_counter()
    vcol, ecol = count_collisions(paths)
    makespan = max(len(p) - 1 for p in paths) if paths else 0
    sum_of_costs = sum(len(p) - 1 for p in paths)
    return {
        "method": "cooperative_astar",
        "paths": paths,
        "individual_search_success": all_success,
        "collision_free": (vcol == 0 and ecol == 0),
        "vertex_collisions": vcol,
        "edge_collisions": ecol,
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "nodes_expanded": total_nodes,
        "time_sec": t1 - t0,
    }
