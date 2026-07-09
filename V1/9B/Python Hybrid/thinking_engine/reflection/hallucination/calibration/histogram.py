# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
HistogramCalibrator
======================
The simplest non-parametric calibrator: bins raw scores into fixed-
width buckets and reports the empirical accuracy rate observed within
each bucket, with no smoothing/prior — in contrast to BetaCalibrator
(same bucketing idea but with a Beta(1,1) prior for graceful sparse-
data behaviour). Useful as a baseline / sanity-check against the
smoothed methods.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .base import BaseCalibrator

_N_BINS = 10


def _bin_index(raw_score: float) -> int:
    return min(int(raw_score * _N_BINS), _N_BINS - 1)


class HistogramCalibrator(BaseCalibrator):
    method_name = "histogram"

    def __init__(self) -> None:
        super().__init__()
        self._bin_counts: Dict[int, List[int]] = {b: [0, 0] for b in range(_N_BINS)}  # [n_total, n_correct]

    def _fit(self) -> None:
        self._bin_counts = {b: [0, 0] for b in range(_N_BINS)}
        for point in self._history:
            b = _bin_index(point.raw_score)
            self._bin_counts[b][0] += 1
            self._bin_counts[b][1] += point.label

    def _predict(self, raw_score: float) -> float:
        b = _bin_index(raw_score)
        total, correct = self._bin_counts[b]
        if total == 0:
            return raw_score  # no data in this bin — fall back to raw score
        return correct / total

    def _export_params(self) -> dict:
        return {"bins": {b: list(v) for b, v in self._bin_counts.items()}}
