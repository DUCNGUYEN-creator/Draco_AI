# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""A* over a generic thought-graph adjacency, with weight-inverted cost
(prefers strongly-related edges) — same fix as engine_v1.py's ASTAR-FIX."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ...knowledge.graph_search import astar as _astar


def astar_search(adj: Dict[str, Dict[str, float]], src: str, dst: str) -> Tuple[Optional[List[str]], float]:
    return _astar(adj, src, dst)
