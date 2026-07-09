# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConsistencyAnalyzer
======================
Rolls up ConsistencyVerifier results across all claims in a response
into one overall "internal coherence" score for the response as a
whole — distinct from reflection.consistency.ConsistencyChecker (which
compares the WHOLE answer text to the reasoning plan) and from
ConsistencyVerifier itself (per-claim, against multiple paths).
"""

from __future__ import annotations

from typing import List


class ConsistencyAnalyzer:
    def overall_coherence(self, per_claim_results: List[List[dict]]) -> float:
        consistency_scores = [
            r.get("score", 0.6)
            for claim_results in per_claim_results
            for r in claim_results
            if r.get("verifier") == "consistency" and r.get("confidence", 0.0) > 0.1
        ]
        if not consistency_scores:
            return 0.6  # neutral default — matches ConsistencyVerifier's own no-signal score
        return sum(consistency_scores) / len(consistency_scores)
