# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BayesianFusion
=================
Sequentially applies each signal as a Bayesian update to a prior belief
that the claim is a hallucination — starting from a NEUTRAL 0.5 prior
and updating once per verifier, treating the verifier's failure
probability as the likelihood ratio input. Differs from NoisyOrFusion
(which assumes signals point toward failure independently, OR-style)
by instead modeling each verifier opinion as EVIDENCE that shifts a
single belief, which handles agreeing "this is FINE" signals
(probability < 0.5) more naturally — noisy-OR can only push risk UP,
never meaningfully down, since it's defined as P(at least one failure).
"""

from __future__ import annotations

from typing import List, Tuple

from ....utils.math import clamp
from ....utils.probability import bayes_update
from ..models.fusion import FusionResult
from .base import BaseFusionStrategy


class BayesianFusion(BaseFusionStrategy):
    method_name = "bayesian"

    def fuse(self, signals: List[Tuple[str, float, float]]) -> FusionResult:
        if not signals:
            return self._empty_result()

        belief = 0.5
        contributions = {}
        for name, p, w in signals:
            # Treat p as a per-verifier "likelihood of failure", scaled
            # toward neutral (0.5) by weight w — a low-weight (heavily
            # discounted/correlated) signal barely moves the belief.
            p_scaled = 0.5 + w * (p - 0.5)
            likelihood_h = clamp(p_scaled, 0.05, 0.95)
            likelihood_nh = clamp(1.0 - p_scaled, 0.05, 0.95)
            before = belief
            belief = bayes_update(belief, likelihood_h, likelihood_nh)
            contributions[name] = round(belief - before, 4)

        return FusionResult(
            method=self.method_name,
            fused_score=belief,
            per_verifier_contribution=contributions,
            n_signals_used=len(signals),
        )
