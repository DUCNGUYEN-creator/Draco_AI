# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""BFS over a generic thought-graph adjacency — reuses knowledge.graph_search.bfs
so reasoning and knowledge never maintain two divergent BFS implementations."""

from __future__ import annotations

from typing import Dict, List, Optional

from ...knowledge.graph_search import bfs as _bfs


def bfs_search(adj: Dict[str, Dict[str, float]], src: str, dst: str) -> Optional[List[str]]:
    return _bfs(adj, src, dst)
