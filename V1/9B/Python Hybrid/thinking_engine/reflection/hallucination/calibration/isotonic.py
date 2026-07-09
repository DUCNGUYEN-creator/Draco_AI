# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
IsotonicCalibrator
=====================
Non-parametric, monotonic step-function calibration via the
Pool-Adjacent-Violators Algorithm (PAVA). More flexible than Platt
(can capture non-sigmoid miscalibration shapes) at the cost of needing
more samples to be reliable — selector.py accounts for this when
choosing between methods.
"""

from __future__ import annotations

from typing import List, Tuple

from .base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    method_name = "isotonic"
    min_samples_to_fit = 10  # PAVA needs more data than Platt to be stable

    def __init__(self) -> None:
        super().__init__()
        self._xs: List[float] = []
        self._ys: List[float] = []  # monotonic step values aligned with _xs

    def _pava(self, points: List[Tuple[float, int]]) -> Tuple[List[float], List[float]]:
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        # Each "block" starts as a single point's label; we merge
        # adjacent blocks whenever monotonicity (non-decreasing) is violated.
        block_values = [float(p[1]) for p in points]
        block_weights = [1.0 for _ in points]
        block_starts = list(range(len(points)))  # index into xs where each block begins

        i = 0
        values, weights, starts = block_values, block_weights, block_starts
        while i < len(values) - 1:
            if values[i] > values[i + 1]:
                merged_weight = weights[i] + weights[i + 1]
                merged_value = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / merged_weight
                values[i : i + 2] = [merged_value]
                weights[i : i + 2] = [merged_weight]
                starts[i : i + 2] = [starts[i]]
                if i > 0:
                    i -= 1
            else:
                i += 1

        # Expand blocks back out to per-point calibrated values.
        ys = []
        for idx, val in enumerate(values):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(xs)
            ys.extend([val] * (end - starts[idx]))
        return xs, ys

    def _fit(self) -> None:
        points = [(p.raw_score, p.label) for p in self._history]
        self._xs, self._ys = self._pava(points)

    def _predict(self, raw_score: float) -> float:
        if not self._xs:
            return raw_score
        # Step-function lookup: find the nearest xs <= raw_score, else
        # extrapolate flatly from the nearest endpoint.
        if raw_score <= self._xs[0]:
            return self._ys[0]
        if raw_score >= self._xs[-1]:
            return self._ys[-1]
        for i in range(len(self._xs) - 1):
            if self._xs[i] <= raw_score <= self._xs[i + 1]:
                # Linear interpolation within the bracketing pair for smoothness.
                x0, x1 = self._xs[i], self._xs[i + 1]
                y0, y1 = self._ys[i], self._ys[i + 1]
                if x1 == x0:
                    return y0
                t = (raw_score - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return self._ys[-1]

    def _export_params(self) -> dict:
        return {"xs": list(self._xs), "ys": list(self._ys)}
