# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
FusionEnsemble
=================
Runs several fusion strategies on the SAME signal set and averages
their fused_scores — hedges against any one method's specific
assumption (independence for noisy-OR, sigmoid-shape for logistic,
2-frame simplification for Dempster-Shafer) being a poor fit for a
particular claim's evidence pattern.
"""

from __future__ import annotations

from typing import List, Tuple

from ..models.fusion import FusionResult
from .base import BaseFusionStrategy
from .logistic import LogisticFusion
from .noisy_or import NoisyOrFusion
from .weighted import WeightedAverageFusion


class FusionEnsemble(BaseFusionStrategy):
    method_name = "fusion_ensemble"

    def __init__(self, members: List[BaseFusionStrategy] | None = None) -> None:
        self.members = members or [NoisyOrFusion(), WeightedAverageFusion(), LogisticFusion()]

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()
        sub_results = [m.fuse(signals) for m in self.members]
        avg_score = sum(r.fused_score for r in sub_results) / len(sub_results)

        merged_contributions = {}
        for r in sub_results:
            for name, contrib in r.per_verifier_contribution.items():
                merged_contributions.setdefault(name, []).append(contrib)
        averaged_contributions = {
            name: round(sum(vals) / len(vals), 4) for name, vals in merged_contributions.items()
        }

        return FusionResult(
            method=self.method_name,
            fused_score=avg_score,
            per_verifier_contribution=averaged_contributions,
            n_signals_used=len(signals),
            notes=[f"sub_methods={[m.method_name for m in self.members]}"],
        )
