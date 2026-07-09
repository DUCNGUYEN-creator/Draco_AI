# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConfidenceScorer
===================
Combines base_confidence (from IntentDetector/DifficultyScorer),
UncertaintyQuantifier's hedge-word analysis, and the Hallucination
report's risk_score into ONE final confidence number — the last step
before ConfidenceCalibrator's learned Platt-scaling correction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ConfidenceScorer:
    def combine(
        self,
        base_confidence: float,
        uncertainty_confidence: Optional[float] = None,
        hallucination_risk_score: Optional[float] = None,
        cot_verification_score: Optional[float] = None,
    ) -> float:
        score = base_confidence
        weights_used = 1.0
        total = score

        if uncertainty_confidence is not None:
            total += uncertainty_confidence
            weights_used += 1.0
        if cot_verification_score is not None:
            total += cot_verification_score
            weights_used += 1.0

        blended = total / weights_used

        if hallucination_risk_score is not None:
            # Risk directly discounts confidence: high risk => lower confidence.
            blended *= (1.0 - 0.6 * hallucination_risk_score)

        return round(max(0.0, min(1.0, blended)), 3)
