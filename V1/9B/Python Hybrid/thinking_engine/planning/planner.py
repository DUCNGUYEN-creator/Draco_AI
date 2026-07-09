# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Planner
=========
Top-level Planning-stage facade composing GoalDecomposer (long-horizon
goals) and PlanDecomposer (immediate ordered steps) behind the single
``interfaces.planner.Planner`` contract. The pipeline's Planning stage
calls this instead of choosing between the two decomposers itself.
"""

from __future__ import annotations

from typing import Any, List

from ..constants import INTENT_CHAT
from ..reasoning.search.mcts import MCTSLight
from .goal_decomposer import GoalDecomposer
from .plan_decomposer import PlanDecomposer

_GOAL_KEYWORDS = ["kế hoạch", "plan", "lộ trình", "roadmap", "30 ngày", "schedule"]


class Planner:
    def __init__(self) -> None:
        self.goal_decomposer = GoalDecomposer()
        self.plan_decomposer = PlanDecomposer(MCTSLight(n_sim=5, max_rollout_depth=8))

    def is_long_horizon_goal(self, question: str) -> bool:
        ql = question.lower()
        return any(kw in ql for kw in _GOAL_KEYWORDS)

    def decompose(
        self,
        goal: str,
        intent: dict,
        max_steps: int = 6,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> List[str]:
        if self.is_long_horizon_goal(goal):
            return self.goal_decomposer.decompose(goal, intent, max_steps, bridge, tokenizer)
        return self.plan_decomposer.decompose(goal, intent, min(max_steps, 4), bridge, tokenizer)
