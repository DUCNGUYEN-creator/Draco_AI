# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
StatisticsAnalyzer
=====================
Computes basic descriptive statistics (mean/stddev/min/max) over a set
of verification results — the shared numeric summary multiple other
analyzers (Outlier, VerifierAgreement) and benchmarks/* consume,
avoiding each one re-implementing the same RunningStats accumulation.
"""

from __future__ import annotations

from typing import Dict, List

from ...hallucination.models.statistics import RunningStats


class StatisticsAnalyzer:
    def summarize(self, verification_results: List[dict], field: str = "score") -> Dict[str, float]:
        values = [r.get(field, 0.0) for r in verification_results]
        if not values:
            return {"mean": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        stats = RunningStats()
        for v in values:
            stats.update(v)
        return {
            "mean": round(stats.mean, 4),
            "stddev": round(stats.stddev, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "n": len(values),
        }
