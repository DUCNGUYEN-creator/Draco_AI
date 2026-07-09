# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
StoppingCriteria
===================
Decides when a reasoning loop (e.g. RecursiveReflectionLoop, council
debate rounds) should stop early — consensus reached, budget exhausted,
or no further improvement observed across iterations.
"""

from __future__ import annotations

from typing import Iterable, List

from .budget import ReasoningBudget


class StoppingCriteria:
    def __init__(self, budget: ReasoningBudget | None = None, min_improvement: float = 0.02) -> None:
        self.budget = budget or ReasoningBudget()
        self.min_improvement = min_improvement
        self._score_history: List[float] = []

    def record_score(self, score: float) -> None:
        self._score_history.append(score)

    def should_stop(self, consensus_reached: bool = False) -> bool:
        if consensus_reached:
            return True
        if self.budget.exhausted():
            return True
        if len(self._score_history) >= 2:
            improvement = self._score_history[-1] - self._score_history[-2]
            if improvement < self.min_improvement:
                return True
        return False
