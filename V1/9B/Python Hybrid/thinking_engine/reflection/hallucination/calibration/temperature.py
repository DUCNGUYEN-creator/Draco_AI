# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TemperatureCalibrator
========================
Single-parameter calibration: calibrated = sigmoid(logit(raw) / T).
The simplest possible learned calibrator (1 parameter vs Platt's 2) —
useful as a fast, very-low-data-requirement default, and as a
numerically stable building block other calibrators can fall back to.
"""

from __future__ import annotations

from ....utils.math import clamp, logit, sigmoid
from .base import BaseCalibrator


class TemperatureCalibrator(BaseCalibrator):
    method_name = "temperature"
    min_samples_to_fit = 3  # single parameter — converges with very little data

    def __init__(self, lr: float = 0.05) -> None:
        super().__init__()
        self.lr = lr
        self._t = 1.0  # T=1 means "no change" (identity through logit/sigmoid round-trip)

    def _fit(self) -> None:
        for point in self._history[-30:]:
            z = logit(point.raw_score) / self._t
            pred = sigmoid(z)
            err = pred - point.label
            # d(loss)/dT via chain rule; clamp T to stay positive and bounded.
            grad = -err * logit(point.raw_score) / (self._t ** 2)
            self._t = clamp(self._t - self.lr * grad, 0.1, 10.0)

    def _predict(self, raw_score: float) -> float:
        return sigmoid(logit(raw_score) / self._t)

    def _export_params(self) -> dict:
        return {"temperature": self._t}
