# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConfidenceCalibrator
=======================
Online Platt-scaling calibrator: calibrated = sigmoid(a * raw_conf + b),
fit via 1-epoch online gradient descent on the last 20 (conf, label)
samples. Ported 1:1 from engine_v1.py's ``ConfidenceCalibrator``.

(This is the lightweight, always-on calibrator used directly by the
pipeline's Reflection stage. reflection.hallucination.calibration
implements the FULL calibration-method zoo — Platt/Isotonic/Beta/
Temperature/Histogram — for the deeper, evidence-level calibration that
happens inside the Hallucination assessor.)
"""

from __future__ import annotations

import math
from typing import List, Tuple


class ConfidenceCalibrator:
    def __init__(self) -> None:
        self._history: List[Tuple[float, int]] = []
        self._a = 1.0
        self._b = 0.0

    def record(self, raw_conf: float, is_correct: bool) -> None:
        self._history.append((raw_conf, int(is_correct)))
        if len(self._history) >= 5:
            self._fit()

    def _fit(self) -> None:
        lr = 0.05
        for conf, label in self._history[-20:]:
            pred = self._sigmoid(self._a * conf + self._b)
            err = pred - label
            self._a -= lr * err * conf
            self._b -= lr * err

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))

    def calibrate(self, raw_conf: float) -> float:
        if len(self._history) < 5:
            return raw_conf
        return round(self._sigmoid(self._a * raw_conf + self._b), 3)
