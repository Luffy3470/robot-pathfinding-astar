"""
heuristics.py
Three admissible heuristics for grid path-finding with 4-connected
(N/E/S/W) movement, unit edge cost.

  1. Manhattan distance   - exact for 4-connected grids w/o obstacles;
                             the "textbook" choice.
  2. Euclidean distance   - straight-line distance; admissible but looser
                             (never overestimates, but underestimates more
                             than Manhattan on a 4-connected grid).
  3. Chebyshev distance   - max(|dr|,|dc|); admissible for 8-connected
                             grids, and still a valid (very loose) lower
                             bound for 4-connected grids since it is
                             <= Manhattan distance always.

A fourth, degenerate heuristic (Dijkstra, h=0) is included as a baseline
so the comparison has a "no information" reference point.
"""
import math


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def zero(a, b):
    """h(n) = 0 everywhere -> A* degenerates to Dijkstra's algorithm."""
    return 0.0


HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "dijkstra_baseline": zero,
}
