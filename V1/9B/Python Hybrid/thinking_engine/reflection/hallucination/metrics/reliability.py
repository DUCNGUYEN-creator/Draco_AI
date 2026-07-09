# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReliabilityTracker
=====================
Online accumulator that tracks, PER VERIFIER, how often that verifier's
"this is well-supported" (score > 0.5) calls turned out correct once
ground truth became available (e.g. via user feedback, a later
correction). This is what should drive registry/factory.py's verifier
weighting over time — a verifier that's historically unreliable for a
given claim type should contribute less to fusion, independent of its
self-reported confidence field.
"""

from __future__ import annotations

from typing import Dict

from ..models.statistics import RunningStats


class ReliabilityTracker:
    def __init__(self) -> None:
        self._correct: Dict[str, int] = {}
        self._total: Dict[str, int] = {}

    def record(self, verifier_name: str, predicted_support: bool, was_correct: bool) -> None:
        self._total[verifier_name] = self._total.get(verifier_name, 0) + 1
        if predicted_support == was_correct:
            self._correct[verifier_name] = self._correct.get(verifier_name, 0) + 1

    def reliability(self, verifier_name: str) -> float:
        total = self._total.get(verifier_name, 0)
        if total == 0:
            return 0.5  # no track record yet — neutral prior
        return self._correct.get(verifier_name, 0) / total

    def all_reliabilities(self) -> Dict[str, float]:
        return {name: self.reliability(name) for name in self._total}
