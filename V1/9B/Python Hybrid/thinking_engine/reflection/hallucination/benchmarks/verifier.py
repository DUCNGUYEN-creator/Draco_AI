# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierBenchmark
====================
Evaluates every registered verifier on a labelled test suite —
(claim, evidence, context, expected_failure: bool) — and reports
AUC, Brier score, and per-verifier reliability so we can track
whether a new verifier is actually better than what it replaces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..metrics.brier import brier_score
from ..metrics.roc import auc_from_points, roc_curve_points
from ..registry.verifier_registry import VerifierRegistry

# A hand-crafted mini test-suite to smoke-test the built-in verifiers.
_MINI_SUITE: List[Tuple[str, Any, Dict, bool]] = [
    ("2 + 2 = 4",  None, {}, False),       # correct — not a hallucination
    ("2 + 2 = 5",  None, {}, True),        # wrong arithmetic — IS a hallucination
    ("A or not A is always true", None, {}, False),   # tautology — true claim
    ("A and not A is always true", None, {}, True),   # contradiction — false claim
]


class VerifierBenchmark:
    def __init__(self, registry: VerifierRegistry | None = None) -> None:
        self.registry = registry or VerifierRegistry()

    def run(self, suite: List[Tuple] | None = None) -> Dict[str, Dict]:
        suite = suite or _MINI_SUITE
        results = {}
        for name in self.registry.available():
            verifier = self.registry.create(name)
            preds, labels = [], []
            for claim, evidence, ctx, is_hallucination in suite:
                try:
                    r = verifier.verify(claim, evidence, ctx)
                    failure_prob = (1.0 - r.get("score", 0.5)) * r.get("confidence", 0.5) + \
                                   0.5 * (1.0 - r.get("confidence", 0.5))
                    preds.append(failure_prob)
                    labels.append(int(is_hallucination))
                except Exception:
                    preds.append(0.5)
                    labels.append(int(is_hallucination))
            if preds:
                pts = roc_curve_points(preds, labels)
                results[name] = {
                    "auc": round(auc_from_points(pts), 4),
                    "brier": round(brier_score(preds, labels), 4),
                    "n_samples": len(preds),
                }
        return results
