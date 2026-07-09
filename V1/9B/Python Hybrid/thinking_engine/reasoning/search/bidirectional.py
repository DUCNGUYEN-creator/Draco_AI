# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Bidirectional Search
=======================
New addition. Runs BFS simultaneously from src and dst, meeting in the
middle — O(b^(d/2)) instead of O(b^d), useful for long knowledge-graph
paths where plain BFS from engine_v1.py's KnowledgeGraph would be slow.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional


def bidirectional_search(
    adj: Dict[str, Dict[str, float]], src: str, dst: str
) -> Optional[List[str]]:
    if src not in adj or dst not in adj:
        return None
    if src == dst:
        return [src]

    front_parent: Dict[str, Optional[str]] = {src: None}
    back_parent: Dict[str, Optional[str]] = {dst: None}
    front_q = deque([src])
    back_q = deque([dst])

    def extend(q: deque, parents: Dict[str, Optional[str]], other_parents: Dict[str, Optional[str]]) -> Optional[str]:
        for _ in range(len(q)):
            node = q.popleft()
            for nb in adj.get(node, {}):
                if nb not in parents:
                    parents[nb] = node
                    if nb in other_parents:
                        return nb
                    q.append(nb)
        return None

    meet = None
    while front_q and back_q and meet is None:
        meet = extend(front_q, front_parent, back_parent)
        if meet is None:
            meet = extend(back_q, back_parent, front_parent)

    if meet is None:
        return None

    # Reconstruct front half: src -> meet
    front_path = [meet]
    node = front_parent[meet]
    while node is not None:
        front_path.append(node)
        node = front_parent[node]
    front_path.reverse()

    # Reconstruct back half: meet -> dst
    back_path = []
    node = back_parent[meet]
    while node is not None:
        back_path.append(node)
        node = back_parent[node]

    return front_path + back_path
