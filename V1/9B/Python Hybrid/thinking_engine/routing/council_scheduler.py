# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CouncilScheduler
==================
Decides how many debate rounds + which experts participate in a given
council/debate session, balancing quality against latency. Wraps
ExpertSelector + a round-count heuristic so reasoning/debate/council.py
doesn't need to know about routing internals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .expert_selector import ExpertSelector


class CouncilScheduler:
    def __init__(self, selector: ExpertSelector | None = None) -> None:
        self.selector = selector or ExpertSelector()

    def schedule(
        self,
        expert_boost: Dict[int, float],
        max_experts: int = 4,
        think_mode: bool = False,
        max_rounds: int = 3,
    ) -> Tuple[List[int], int]:
        experts = self.selector.select(expert_boost, max_experts)
        rounds = max_rounds if think_mode else min(2, max_rounds)
        return experts, rounds
