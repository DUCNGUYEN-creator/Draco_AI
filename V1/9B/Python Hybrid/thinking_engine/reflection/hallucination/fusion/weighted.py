# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
WeightedAverageFusion
========================
The simplest, most interpretable fusion method: weighted mean of
per-verifier failure probabilities. Use when signals are roughly
independent and similarly scaled — selector.py's "fast" strategy
default, since it's O(n) and trivially explainable in a report.
"""

from __future__ import annotations

from typing import List, Tuple

from ..models.fusion import FusionResult
from .base import BaseFusionStrategy


class WeightedAverageFusion(BaseFusionStrategy):
    method_name = "weighted"

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()
        total_weight = sum(w for _, _, w in signals)
        if total_weight <= 0:
            return self._empty_result()
        fused = sum(p * w for _, p, w in signals) / total_weight
        contributions = {name: round((p * w) / total_weight, 4) for name, p, w in signals}
        return FusionResult(
            method=self.method_name,
            fused_score=fused,
            per_verifier_contribution=contributions,
            n_signals_used=len(signals),
        )
