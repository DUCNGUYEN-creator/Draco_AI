# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CorrelationPipeline — Stage 4: "Correlation"
================================================
Applies correlation.reducer.EvidenceReducer's verifier-discount logic
to the raw verification results, producing per-verifier WEIGHT
multipliers that FusionPipeline uses alongside each verifier's own
confidence — this is what prevents structurally-dependent verifier
pairs (numerical+tool, symbolic+reasoning, retrieval+citation) from
being double-counted as independent confirmations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..correlation.reducer import EvidenceReducer


class CorrelationPipeline:
    def __init__(self, reducer: Optional[EvidenceReducer] = None) -> None:
        self.reducer = reducer or EvidenceReducer()

    def compute_weights(self, verification_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Returns {verifier_name: discount_multiplier in (0, 1]}."""
        return self.reducer.verifier_discounts(verification_results)

    def to_fusion_signals(self, verification_results: List[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
        """Converts raw VerificationResult dicts into the (name,
        failure_probability, weight) triples fusion/*.py expects —
        weight = verifier's own confidence * its correlation discount."""
        discounts = self.compute_weights(verification_results)
        signals: List[Tuple[str, float, float]] = []
        for r in verification_results:
            name = r.get("verifier", "unknown")
            score = r.get("score", 0.5)
            confidence = r.get("confidence", 0.0)
            if confidence <= 0.0:
                continue  # abstaining verifier contributes no signal
            failure_prob = (1.0 - score) * confidence + 0.5 * (1.0 - confidence)
            weight = confidence * discounts.get(name, 1.0)
            signals.append((name, failure_prob, weight))
        return signals
