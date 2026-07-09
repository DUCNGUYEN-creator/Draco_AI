# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
MCTSLight / MCTSNode
=======================
Lightweight Monte-Carlo Tree Search over a flat list of candidate
"thought" branches (not a full game tree) — exactly the shape
engine_v1.py used to pick the best branch out of Tree-of-Thoughts,
GoalDecomposer, PlanDecomposer, and AbductionEngine candidates. Ported
1:1, including the max_rollout_depth=10 default that prevents infinite
expansion.
"""

from __future__ import annotations

import math
from typing import List, Optional


class MCTSNode:
    def __init__(self, thought: str, parent: Optional["MCTSNode"] = None) -> None:
        self.thought = thought
        self.parent = parent
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.score = 0.0

    def uct(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        return self.score / self.visits + c * math.sqrt(
            math.log(self.parent.visits + 1) / self.visits
        )

    def best_child(self) -> "MCTSNode":
        return max(self.children, key=lambda n: n.uct())


class MCTSLight:
    def __init__(self, n_sim: int = 10, max_rollout_depth: int = 10) -> None:
        self.n_sim = n_sim
        self.max_rollout_depth = max_rollout_depth

    def search(self, question: str, branches: List[str]) -> str:
        if not branches:
            return ""
        root = MCTSNode(f"Q: {question}")
        for b in branches:
            root.children.append(MCTSNode(b, root))
        for _ in range(self.n_sim):
            node = self._select(root)
            score = self._simulate(node.thought, question)
            self._backprop(node, score)
        return max(root.children, key=lambda n: n.score / max(n.visits, 1)).thought

    def _select(self, node: MCTSNode) -> MCTSNode:
        depth = 0
        while node.children and depth < self.max_rollout_depth:
            node = node.best_child()
            depth += 1
        return node

    def _simulate(self, thought: str, question: str) -> float:
        score = 0.5
        score += min(len(thought) / 200.0, 0.2)
        q_w = set(question.lower().split())
        t_w = set(thought.lower().split())
        score += min(len(q_w & t_w) * 0.05, 0.3)
        return min(score, 1.0)

    def _backprop(self, node: MCTSNode, score: float) -> None:
        while node:
            node.visits += 1
            node.score += score
            node = node.parent
