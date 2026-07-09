# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BestOfN
=========
New addition. Generates N independent candidate thoughts (via a
caller-supplied generator function) and scores each with a
caller-supplied scorer, returning the single best one. Distinct from
SelfConsistency (which votes by keyword overlap across MCTS-derived
paths): BestOfN is agnostic to how candidates are produced or scored,
making it the right tool when an external reward model is plugged in.
"""

from __future__ import annotations

from typing import Callable, List, Tuple


class BestOfN:
    def run(
        self,
        n: int,
        generate_fn: Callable[[int], str],
        score_fn: Callable[[str], float],
    ) -> Tuple[str, List[Tuple[str, float]]]:
        candidates = [generate_fn(i) for i in range(n)]
        scored = [(c, score_fn(c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0] if scored else ""
        return best, scored
