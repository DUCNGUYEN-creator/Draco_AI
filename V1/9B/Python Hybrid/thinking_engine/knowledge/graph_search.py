# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
graph_search
==============
BFS / DFS / A* traversal over the {node: {neighbor: weight}} adjacency
shape used by KnowledgeGraph. Ported 1:1 from the methods embedded in
engine_v1.py's ``KnowledgeGraph`` (BFS popleft fix, A* cost inversion
fix, src/dst-not-in-graph guards). Split out as free functions so
reasoning/search/* can reuse the same algorithms over other adjacency
structures without subclassing KnowledgeGraph.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

Adjacency = Dict[str, Dict[str, float]]

_EPS = 1e-6


def bfs(adj: Adjacency, src: str, dst: str) -> Optional[List[str]]:
    if src not in adj or dst not in adj:
        return None
    if src == dst:
        return [src]
    q = deque([[src]])
    vis = {src}
    while q:
        path = q.popleft()
        for nb in adj.get(path[-1], {}):
            if nb in vis:
                continue
            new_path = path + [nb]
            if nb == dst:
                return new_path
            vis.add(nb)
            q.append(new_path)
    return None


def dfs(adj: Adjacency, src: str, dst: str, max_d: int = 6) -> Optional[List[str]]:
    if src not in adj or dst not in adj:
        return None
    stack = [(src, [src])]
    vis: set = set()
    while stack:
        node, path = stack.pop()
        if node == dst:
            return path
        if node in vis or len(path) > max_d:
            continue
        vis.add(node)
        for nb in adj.get(node, {}):
            if nb not in vis:
                stack.append((nb, path + [nb]))
    return None


def astar(adj: Adjacency, src: str, dst: str) -> Tuple[Optional[List[str]], float]:
    """heuristic = 0.0 if same node else 1.0 (admissible).
    Cost inversion: KG stores semantic-similarity weights (high = stronger
    link); A* minimises cost, so cost = 1 / (w + eps) — this correctly
    prefers high-weight (strongly related) edges."""
    if src not in adj or dst not in adj:
        return None, math.inf
    h = lambda a, b: 0.0 if a == b else 1.0
    heap = [(0.0, 0.0, src, [src])]
    gs = defaultdict(lambda: math.inf)
    gs[src] = 0.0
    while heap:
        f, g, node, path = heapq.heappop(heap)
        if node == dst:
            return path, g
        if g > gs[node]:
            continue
        for nb, weight in adj.get(node, {}).items():
            edge_cost = 1.0 / (weight + _EPS)
            ng = g + edge_cost
            if ng < gs[nb]:
                gs[nb] = ng
                heapq.heappush(heap, (ng + h(nb, dst), ng, nb, path + [nb]))
    return None, math.inf
