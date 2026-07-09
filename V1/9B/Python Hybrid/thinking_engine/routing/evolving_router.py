# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SelfEvolvingRouter
=====================
Thompson-Sampling-style online update of per-(intent, expert) Beta
distribution parameters. Ported 1:1 from engine_v1.py's
``SelfEvolvingRouter``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple


class SelfEvolvingRouter:
    """
    update(intent_type, expert_id, success=True/False)
    apply(intent_type, intent_boost) -> adjusted_boost
    """

    def __init__(self) -> None:
        self._alpha: Dict[Tuple[str, int], float] = defaultdict(lambda: 1.0)
        self._beta: Dict[Tuple[str, int], float] = defaultdict(lambda: 1.0)

    def update(self, intent_type: str, expert_id: int, success: bool) -> None:
        k = (intent_type, expert_id)
        if success:
            self._alpha[k] += 1.0
        else:
            self._beta[k] += 1.0

    def apply(self, intent_type: str, intent_boost: Dict[int, float]) -> Dict[int, float]:
        """Draw Thompson-sample means and blend 50/50 with intent_boost."""
        sampled: Dict[int, float] = {}
        for eid in intent_boost:
            k = (intent_type, eid)
            a = self._alpha[k]
            b = self._beta[k]
            mu = a / (a + b)
            sampled[eid] = mu

        blended: Dict[int, float] = {}
        for eid in intent_boost:
            blended[eid] = 0.5 * intent_boost[eid] + 0.5 * sampled.get(eid, 0.5)

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended
