# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
OutlierAnalyzer
==================
Detects when ONE verifier's score is a statistical outlier relative to
the rest of the ensemble for the same claim — e.g. 5 verifiers score
~0.8 and one scores 0.1. Outliers are flagged (not silently dropped):
they might be the single verifier that caught a real, subtle problem
everything else missed, or they might be a misconfigured/buggy
verifier. fusion/* decides what to do with the flag; this analyzer
only detects and reports it.
"""

from __future__ import annotations

from typing import Dict, List

from ...hallucination.models.statistics import RunningStats

_OUTLIER_Z_THRESHOLD = 1.5


class OutlierAnalyzer:
    def find_outliers(self, verification_results: List[dict]) -> List[Dict[str, object]]:
        informative = [r for r in verification_results if r.get("confidence", 0.0) > 0.1]
        if len(informative) < 3:
            return []  # not enough signals for outlier detection to be meaningful

        stats = RunningStats()
        for r in informative:
            stats.update(r["score"])
        if stats.stddev == 0:
            return []

        outliers = []
        for r in informative:
            z = abs(r["score"] - stats.mean) / stats.stddev
            if z >= _OUTLIER_Z_THRESHOLD:
                outliers.append({"verifier": r.get("verifier"), "score": r["score"], "z_score": round(z, 3)})
        return outliers
