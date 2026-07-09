# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PlattCalibrator
==================
Logistic (Platt) scaling: calibrated = sigmoid(a * raw + b), fit via
online gradient descent over the last 50 samples (deeper history window
than reflection.confidence_calibrator's lightweight 20-sample version,
since this calibrator is meant for the higher-stakes, less-frequent
final risk-score calibration rather than the always-on per-turn
confidence calibration).
"""

from __future__ import annotations

from ....utils.math import clamp, sigmoid
from .base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    method_name = "platt"

    def __init__(self, lr: float = 0.05, window: int = 50) -> None:
        super().__init__()
        self.lr = lr
        self.window = window
        self._a = 1.0
        self._b = 0.0

    def _fit(self) -> None:
        for point in self._history[-self.window :]:
            pred = sigmoid(self._a * point.raw_score + self._b)
            err = pred - point.label
            self._a -= self.lr * err * point.raw_score
            self._b -= self.lr * err

    def _predict(self, raw_score: float) -> float:
        return sigmoid(self._a * raw_score + self._b)

    def _export_params(self) -> dict:
        return {"a": self._a, "b": self._b}
