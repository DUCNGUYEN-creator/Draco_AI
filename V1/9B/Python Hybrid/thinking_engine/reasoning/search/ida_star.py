# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
IDA* (Iterative Deepening A*)
================================
New addition. Memory-efficient alternative to A* for very large
thought-graphs: repeatedly runs depth-limited DFS with an increasing
cost threshold, avoiding A*'s open-set memory blow-up. Uses the same
weight-inversion cost convention as knowledge.graph_search.astar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

_EPS = 1e-6


def ida_star_search(
    adj: Dict[str, Dict[str, float]], src: str, dst: str, max_threshold_iters: int = 50
) -> Tuple[Optional[List[str]], float]:
    if src not in adj or dst not in adj:
        return None, float("inf")

    def h(node: str) -> float:
        return 0.0 if node == dst else 1.0

    def edge_cost(w: float) -> float:
        return 1.0 / (w + _EPS)

    threshold = h(src)
    path = [src]

    def search(g: float, bound: float) -> Tuple[float, bool]:
        node = path[-1]
        f = g + h(node)
        if f > bound:
            return f, False
        if node == dst:
            return f, True
        min_exceed = float("inf")
        for nb, w in adj.get(node, {}).items():
            if nb in path:
                continue
            path.append(nb)
            t, found = search(g + edge_cost(w), bound)
            if found:
                return t, True
            if t < min_exceed:
                min_exceed = t
            path.pop()
        return min_exceed, False

    for _ in range(max_threshold_iters):
        t, found = search(0.0, threshold)
        if found:
            total_cost = sum(
                edge_cost(adj[path[i]][path[i + 1]]) for i in range(len(path) - 1)
            )
            return list(path), total_cost
        if t == float("inf"):
            return None, float("inf")
        threshold = t
    return None, float("inf")
