# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierScoreTracker
=======================
Accumulates RunningStats of a verifier's raw score output over time,
per verifier name — the input drift.py's DriftDetector consumes to spot
when a verifier's typical output distribution has shifted (e.g. after a
KG schema change made RetrievalVerifier systematically more pessimistic).
"""

from __future__ import annotations

from typing import Dict

from ..models.statistics import RunningStats


class VerifierScoreTracker:
    def __init__(self) -> None:
        self._stats: Dict[str, RunningStats] = {}

    def record(self, verifier_name: str, score: float) -> None:
        self._stats.setdefault(verifier_name, RunningStats()).update(score)

    def get_stats(self, verifier_name: str) -> RunningStats:
        return self._stats.setdefault(verifier_name, RunningStats())

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {"n": s.n, "mean": round(s.mean, 4), "stddev": round(s.stddev, 4)}
            for name, s in self._stats.items()
        }
