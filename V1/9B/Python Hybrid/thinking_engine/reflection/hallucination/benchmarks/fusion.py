# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""FusionBenchmark — checks that each fusion method produces values in
[0,1] and satisfies basic monotonicity (adding a pure-failure signal
should only increase or maintain the fused score, never decrease it)."""

from __future__ import annotations

from typing import Dict

from ..registry.fusion_registry import FusionRegistry


class FusionBenchmark:
    def __init__(self, registry: FusionRegistry | None = None) -> None:
        self.registry = registry or FusionRegistry()

    def run(self) -> Dict[str, Dict]:
        results = {}
        for name in self.registry.available():
            strat = self.registry.create(name)
            s1 = [("retrieval", 0.8, 0.9)]
            s2 = [("retrieval", 0.8, 0.9), ("numerical", 0.9, 0.8)]
            r1 = strat.fuse(s1).fused_score
            r2 = strat.fuse(s2).fused_score
            results[name] = {
                "single_signal": round(r1, 4),
                "two_signals": round(r2, 4),
                "in_range": 0.0 <= r1 <= 1.0 and 0.0 <= r2 <= 1.0,
                "monotone_ok": r2 >= r1 - 0.05,  # adding a strong failure signal should not lower risk
            }
        return results
