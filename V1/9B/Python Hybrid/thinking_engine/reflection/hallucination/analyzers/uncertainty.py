# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
UncertaintyAnalyzer
=====================
Computes the SHANNON ENTROPY of the verifier-confidence distribution
for a claim — high entropy (many verifiers all somewhat-but-not-very
confident) is a different failure mode than low entropy with low mean
confidence (verifiers all confidently disagree, or all confidently
abstain). Feeds calibration/* as an auxiliary feature beyond the raw
fused score.
"""

from __future__ import annotations

from typing import Dict, List

from ...hallucination.models.statistics import RunningStats
from ....utils.math import entropy as _entropy
from ....utils.math import softmax as _softmax


class UncertaintyAnalyzer:
    def analyze(self, verification_results: List[dict]) -> Dict[str, float]:
        confidences = [r.get("confidence", 0.0) for r in verification_results]
        if not confidences:
            return {"entropy": 0.0, "mean_confidence": 0.0}
        probs = _softmax(confidences) if sum(confidences) > 0 else [1.0 / len(confidences)] * len(confidences)
        ent = _entropy(probs)
        stats = RunningStats()
        for c in confidences:
            stats.update(c)
        return {"entropy": round(ent, 4), "mean_confidence": round(stats.mean, 4)}
