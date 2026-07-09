# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
NoisyOrFusion
===============
P(hallucination) = 1 - prod(1 - w_i * p_i) over all signals — the
correct combination rule when ANY ONE strong, confident "this is
wrong" signal should dominate the fused risk, rather than being
averaged away by several weak/neutral verifiers (WeightedAverageFusion's
failure mode: one verifier screaming 0.95 risk gets diluted to ~0.3 by
five abstaining verifiers near 0.5). This is config.HallucinationConfig's
DEFAULT fusion method for exactly this reason — false negatives
(missing a real hallucination) are usually costlier than false
positives in this domain.
"""

from __future__ import annotations

from typing import List, Tuple

from ....utils.probability import noisy_or
from ..models.fusion import FusionResult
from .base import BaseFusionStrategy


class NoisyOrFusion(BaseFusionStrategy):
    method_name = "noisy_or"

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()
        weighted_probs = [p * w for _, p, w in signals]
        fused = noisy_or(weighted_probs)

        # Contribution = how much each signal alone would have produced,
        # normalized so contributions are comparable/interpretable in a report
        # (they won't sum to fused_score exactly — noisy-OR isn't additive —
        # but the RELATIVE ranking across verifiers is still meaningful).
        total = sum(weighted_probs) or 1.0
        contributions = {name: round((p * w) / total, 4) for name, p, w in signals}

        return FusionResult(
            method=self.method_name,
            fused_score=fused,
            per_verifier_contribution=contributions,
            n_signals_used=len(signals),
        )
