# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
GraphOfThoughts
==================
New addition. Generalizes Tree-of-Thoughts: thoughts can merge (two
branches combined into a synthesis node) as well as branch, modeled as
a small DAG instead of a strict tree. Useful when SelfConsistency-style
multiple paths should be cross-pollinated rather than purely voted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..search.mcts import MCTSLight


@dataclass
class ThoughtNode:
    id: str
    text: str
    parents: List[str] = field(default_factory=list)
    score: float = 0.0


class GraphOfThoughts:
    def __init__(self, mcts: Optional[MCTSLight] = None) -> None:
        self.mcts = mcts or MCTSLight(n_sim=6, max_rollout_depth=8)
        self.nodes: Dict[str, ThoughtNode] = {}

    def add_thought(self, node_id: str, text: str, parents: Optional[List[str]] = None) -> ThoughtNode:
        node = ThoughtNode(id=node_id, text=text, parents=parents or [])
        self.nodes[node_id] = node
        return node

    def merge(self, node_ids: List[str], merged_id: str, question: str) -> ThoughtNode:
        """Combine several thought nodes into one synthesis node — picks the
        best phrasing among the originals via MCTS rather than naive
        concatenation (which tends to produce bloated, redundant text)."""
        texts = [self.nodes[nid].text for nid in node_ids if nid in self.nodes]
        if not texts:
            return self.add_thought(merged_id, "", parents=node_ids)
        best = self.mcts.search(question, texts)
        return self.add_thought(merged_id, best, parents=node_ids)

    def topological_order(self) -> List[str]:
        visited: Dict[str, bool] = {}
        order: List[str] = []

        def visit(nid: str) -> None:
            if visited.get(nid):
                return
            visited[nid] = True
            for p in self.nodes[nid].parents:
                if p in self.nodes:
                    visit(p)
            order.append(nid)

        for nid in self.nodes:
            visit(nid)
        return order
