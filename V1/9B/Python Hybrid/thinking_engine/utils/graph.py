# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Generic graph primitives reused by knowledge/graph_search.py,
reasoning/search/*.py and hallucination/correlation/graph.py — avoids
three independent re-implementations of the same BFS/union-find code."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple


def bfs_path(adj: Dict[str, Dict[str, float]], src: str, dst: str) -> Optional[List[str]]:
    if src not in adj or dst not in adj:
        return None
    if src == dst:
        return [src]
    q = deque([[src]])
    visited = {src}
    while q:
        path = q.popleft()
        for nb in adj.get(path[-1], {}):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb == dst:
                return new_path
            visited.add(nb)
            q.append(new_path)
    return None


class UnionFind:
    """Union-Find / Disjoint-Set, used for connected-components clustering
    in hallucination/correlation/connected_components.py."""

    def __init__(self, items: Iterable[str]):
        self._parent = {x: x for x in items}
        self._rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> List[Set[str]]:
        groups: Dict[str, Set[str]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, set()).add(x)
        return list(groups.values())
