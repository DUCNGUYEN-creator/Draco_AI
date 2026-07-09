# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CoverageAnalyzer
===================
Measures what FRACTION of the configured verifier set actually produced
an informative (non-abstaining) opinion on a claim. A claim checked by
only 1 of 8 possible verifiers (most abstained — e.g. no arithmetic, no
citations, no tool results) deserves lower assessment-confidence than
one where 6 of 8 verifiers weighed in, REGARDLESS of what score those
verifiers gave. Strategy/* (fast/balanced/paranoid) uses this to decide
whether to escalate to a deeper verifier set.
"""

from __future__ import annotations

from typing import Dict, List

_ABSTAIN_CONFIDENCE_THRESHOLD = 0.1


class CoverageAnalyzer:
    def coverage(self, verification_results: List[dict], n_configured_verifiers: int) -> Dict[str, float]:
        informative = [r for r in verification_results if r.get("confidence", 0.0) > _ABSTAIN_CONFIDENCE_THRESHOLD]
        n_informative = len(informative)
        ratio = n_informative / n_configured_verifiers if n_configured_verifiers else 0.0
        return {
            "n_configured": n_configured_verifiers,
            "n_informative": n_informative,
            "n_abstained": len(verification_results) - n_informative,
            "coverage_ratio": round(ratio, 4),
        }
