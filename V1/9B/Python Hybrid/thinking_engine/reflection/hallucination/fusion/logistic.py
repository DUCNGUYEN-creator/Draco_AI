# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
LogisticFusion
=================
Combines signals in LOGIT space (sum of weighted logits, then
sigmoid back) rather than probability space — mathematically the
"correct" way to combine independent log-odds, and avoids
WeightedAverageFusion's tendency to compress extreme probabilities
toward the middle (averaging a 0.99 and a 0.5 in probability space
gives 0.745; averaging their logits and converting back gives a result
much closer to 0.95, correctly reflecting that one verifier was nearly
certain).
"""

from __future__ import annotations

from typing import List, Tuple

from ....utils.math import logit, sigmoid
from ..models.fusion import FusionResult
from .base import BaseFusionStrategy


class LogisticFusion(BaseFusionStrategy):
    method_name = "logistic"

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()
        total_weight = sum(w for _, _, w in signals)
        if total_weight <= 0:
            return self._empty_result()

        weighted_logit_sum = sum(w * logit(p) for _, p, w in signals)
        fused_logit = weighted_logit_sum / total_weight
        fused = sigmoid(fused_logit)

        contributions = {
            name: round((w * logit(p)) / total_weight, 4) for name, p, w in signals
        }
        return FusionResult(
            method=self.method_name,
            fused_score=fused,
            per_verifier_contribution=contributions,
            n_signals_used=len(signals),
        )
