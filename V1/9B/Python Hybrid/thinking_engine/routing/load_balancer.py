# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExpertLoadBalancer
=====================
Tracks per-expert usage count and running performance score, applying a
soft equity bonus to under-used experts so they don't starve. Ported
1:1 from engine_v1.py's ``ExpertLoadBalancer``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable


class ExpertLoadBalancer:
    def __init__(self, n_experts: int = 8) -> None:
        self.n_experts = n_experts
        self.usage_count: Dict[int, int] = defaultdict(int)
        self.perf_score: Dict[int, float] = defaultdict(float)
        self.perf_calls: Dict[int, int] = defaultdict(int)

    def balanced_boost(self, intent_boost: Dict[int, float]) -> Dict[int, float]:
        """Blend intent_boost with a small equity bonus for under-used
        experts: equity = 0.05 * (1 - usage_fraction). Re-normalized to
        sum to 1.0."""
        total_usage = max(sum(self.usage_count.values()), 1)
        boosted: Dict[int, float] = {}
        for exp_id in range(self.n_experts):
            usage_frac = self.usage_count[exp_id] / total_usage
            equity = 0.05 * (1.0 - usage_frac)
            boosted[exp_id] = intent_boost.get(exp_id, 0.0) + equity
        total = sum(boosted.values())
        if total > 0:
            boosted = {k: v / total for k, v in boosted.items()}
        return boosted

    def record_usage(self, expert_ids: Iterable[int]) -> None:
        for eid in expert_ids:
            self.usage_count[eid] += 1

    def update_score(self, expert_id: int, rating: float) -> None:
        """Exponential moving average (alpha=0.2). rating in [0.0, 1.0]."""
        alpha = 0.2
        prev = self.perf_score[expert_id]
        n = self.perf_calls[expert_id]
        if n == 0:
            self.perf_score[expert_id] = rating
        else:
            self.perf_score[expert_id] = (1 - alpha) * prev + alpha * rating
        self.perf_calls[expert_id] += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "usage": dict(self.usage_count),
            "perf": {k: round(v, 3) for k, v in self.perf_score.items()},
        }
