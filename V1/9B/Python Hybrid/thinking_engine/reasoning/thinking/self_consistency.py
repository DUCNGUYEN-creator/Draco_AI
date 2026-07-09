# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SelfConsistency
==================
Generates multiple reasoning paths (with branch rotation + randomized
ordering for real diversity) and votes for the path with highest
keyword overlap with the question. Ported 1:1 from engine_v1.py's
``SelfConsistency``.
"""

from __future__ import annotations

import random
from typing import List

from ..search.mcts import MCTSLight
from .tree_of_thought import TreeOfThoughts


class SelfConsistency:
    def generate_paths(self, question: str, intent: dict, n_paths: int = 3) -> List[str]:
        base = TreeOfThoughts(MCTSLight(n_sim=5, max_rollout_depth=8))
        paths = []
        for i in range(n_paths):
            branches = base.generate_branches(question, intent)
            shuffled = branches[:]
            random.shuffle(shuffled)
            rotated = shuffled[i % len(shuffled) :] + shuffled[: i % len(shuffled)]
            best = base.mcts.search(question, rotated)
            paths.append(f"[PATH {i + 1}] {best}")
        return paths

    def vote(self, paths: List[str], question: str) -> str:
        q_words = set(question.lower().split())
        return max(paths, key=lambda p: len(q_words & set(p.lower().split())))
