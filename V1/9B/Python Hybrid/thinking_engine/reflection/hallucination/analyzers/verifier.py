# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierAgreementAnalyzer
============================
Measures how much the individual verifiers in an ensemble AGREE with
each other on a single claim. Low agreement (e.g. RetrievalVerifier
says "well-supported" while ContradictionVerifier says "conflicts")
is itself a hallucination-risk signal distinct from any one verifier's
score — it flags claims where the evidence picture is genuinely murky.
"""

from __future__ import annotations

from typing import Dict, List

from ...hallucination.models.statistics import RunningStats


class VerifierAgreementAnalyzer:
    def analyze(self, verification_results: List[dict]) -> Dict[str, float]:
        scores = [r.get("score", 0.5) for r in verification_results if r.get("confidence", 0.0) > 0.1]
        if len(scores) < 2:
            return {"agreement": 1.0, "n_compared": len(scores), "score_spread": 0.0}

        stats = RunningStats()
        for s in scores:
            stats.update(s)
        spread = stats.stddev
        # Map spread (0 = perfect agreement, ~0.5 = max possible spread for [0,1] scores)
        # to an agreement score in [0, 1].
        agreement = max(0.0, 1.0 - spread * 2.0)
        return {"agreement": round(agreement, 4), "n_compared": len(scores), "score_spread": round(spread, 4)}
