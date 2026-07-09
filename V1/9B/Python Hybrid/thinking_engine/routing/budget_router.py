# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BudgetRouter
==============
New addition: routes a "compute budget" (token/time allowance) to each
reasoning sub-task based on process_mode and difficulty_score, so
expensive sub-systems (Council, MCTS rollouts) get more budget on hard
queries and less on easy ones — preventing the kind of unconditional
8-expert full council engine_v1.py guarded against with max_experts.
"""

from __future__ import annotations

from typing import Dict


class BudgetRouter:
    def allocate(self, process_mode: str, difficulty_score: float) -> Dict[str, int]:
        """Returns a dict of per-subsystem step/round budgets."""
        if process_mode == "slow" or difficulty_score >= 0.65:
            return {
                "mcts_sim": 12,
                "mcts_rollout_depth": 12,
                "debate_rounds": 3,
                "self_consistency_paths": 3,
                "goal_decomposer_depth": 20,
            }
        return {
            "mcts_sim": 6,
            "mcts_rollout_depth": 8,
            "debate_rounds": 1,
            "self_consistency_paths": 1,
            "goal_decomposer_depth": 10,
        }
