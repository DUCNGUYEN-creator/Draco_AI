# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BetaCalibrator
================
Models the calibrated probability as a Beta-Binomial posterior mean:
treats each (raw_score >= 0.5, label) pair as a Bernoulli trial bucketed
by raw_score decile, accumulating alpha/beta per bucket. Naturally
handles sparse data via Beta(1,1) uniform prior — degrades gracefully
to "uninformative 0.5" for buckets with no samples yet, rather than
the discontinuous jumps IsotonicCalibrator can show with few samples
per region.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .base import BaseCalibrator

_N_BUCKETS = 10


def _bucket(raw_score: float) -> int:
    return min(int(raw_score * _N_BUCKETS), _N_BUCKETS - 1)


class BetaCalibrator(BaseCalibrator):
    method_name = "beta"

    def __init__(self) -> None:
        super().__init__()
        self._alpha: Dict[int, float] = {b: 1.0 for b in range(_N_BUCKETS)}
        self._beta: Dict[int, float] = {b: 1.0 for b in range(_N_BUCKETS)}

    def _fit(self) -> None:
        # Recompute from scratch each time (cheap: bucket counts only) —
        # avoids double-counting if record() somehow gets called with
        # the same history twice.
        self._alpha = {b: 1.0 for b in range(_N_BUCKETS)}
        self._beta = {b: 1.0 for b in range(_N_BUCKETS)}
        for point in self._history:
            b = _bucket(point.raw_score)
            if point.label == 1:
                self._alpha[b] += 1.0
            else:
                self._beta[b] += 1.0

    def _predict(self, raw_score: float) -> float:
        b = _bucket(raw_score)
        a, beta_param = self._alpha[b], self._beta[b]
        return a / (a + beta_param)

    def _export_params(self) -> dict:
        return {"alpha": dict(self._alpha), "beta": dict(self._beta)}
