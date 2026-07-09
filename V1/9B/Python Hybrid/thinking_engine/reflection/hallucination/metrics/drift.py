# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DriftDetector
================
Flags when a verifier's recent-window mean score has shifted
significantly from its long-run baseline — a Welford-friendly two-
window comparison (no need to store raw history). telemetry.py polls
this periodically; it is NOT consulted in the per-request hot path.
"""

from __future__ import annotations

from typing import Dict

from ..models.statistics import RunningStats

_DEFAULT_DRIFT_THRESHOLD = 0.15  # absolute mean-shift considered "drift"


class DriftDetector:
    """Maintains two genuinely separate windows per verifier:
    - ``_baseline``: a RunningStats accumulator over every score that has
      ALREADY rolled out of the recent window (i.e. established history).
    - ``_recent``: a bounded FIFO of the last ``recent_window`` scores.

    A new score always enters ``_recent`` first; only when it is evicted
    from the recent window (because the window is full) does it get
    folded into ``_baseline``. This guarantees the two windows never
    overlap, so a sudden shift in the recent window is compared against
    a baseline that genuinely predates it — unlike accumulating
    baseline over the full history, which would let the recent window's
    own scores leak into and dilute the baseline it's being compared
    against, masking real drift.
    """

    def __init__(self, threshold: float = _DEFAULT_DRIFT_THRESHOLD, recent_window: int = 50) -> None:
        self.threshold = threshold
        self.recent_window = recent_window
        self._baseline: Dict[str, RunningStats] = {}
        self._recent: Dict[str, list] = {}

    def record(self, verifier_name: str, score: float) -> None:
        recent = self._recent.setdefault(verifier_name, [])
        recent.append(score)
        if len(recent) > self.recent_window:
            evicted = recent.pop(0)
            self._baseline.setdefault(verifier_name, RunningStats()).update(evicted)

    def check_drift(self, verifier_name: str) -> Dict[str, object]:
        baseline = self._baseline.get(verifier_name)
        recent = self._recent.get(verifier_name, [])
        if baseline is None or baseline.n < 10 or len(recent) < 10:
            return {"drifted": False, "reason": "insufficient_data"}
        recent_mean = sum(recent) / len(recent)
        shift = abs(recent_mean - baseline.mean)
        return {
            "drifted": shift >= self.threshold,
            "baseline_mean": round(baseline.mean, 4),
            "recent_mean": round(recent_mean, 4),
            "shift": round(shift, 4),
        }
