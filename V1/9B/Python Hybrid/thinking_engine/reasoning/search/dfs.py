# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""DFS over a generic thought-graph adjacency, with max-depth guard."""

from __future__ import annotations

from typing import Dict, List, Optional

from ...knowledge.graph_search import dfs as _dfs


def dfs_search(adj: Dict[str, Dict[str, float]], src: str, dst: str, max_d: int = 6) -> Optional[List[str]]:
    return _dfs(adj, src, dst, max_d)
