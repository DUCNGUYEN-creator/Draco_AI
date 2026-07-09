# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationEnsemble
======================
Averages predictions from multiple calibration methods — reduces
variance from any single method's modeling assumptions being wrong for
the current data distribution (e.g. Platt assumes a sigmoid shape;
Isotonic assumes enough samples per region; averaging hedges both
risks). All sub-calibrators receive the SAME (raw_score, label)
stream via record(), so they all fit on identical data.
"""

from __future__ import annotations

from typing import List

from .base import BaseCalibrator
from .beta import BetaCalibrator
from .platt import PlattCalibrator
from .temperature import TemperatureCalibrator


class CalibrationEnsemble:
    method_name = "ensemble"

    def __init__(self, members: List[BaseCalibrator] | None = None) -> None:
        self.members = members or [PlattCalibrator(), TemperatureCalibrator(), BetaCalibrator()]

    def record(self, raw_score: float, label: int) -> None:
        for m in self.members:
            m.record(raw_score, label)

    def calibrate(self, raw_score: float) -> float:
        preds = [m.calibrate(raw_score) for m in self.members]
        return sum(preds) / len(preds) if preds else raw_score

    @property
    def n_samples(self) -> int:
        return self.members[0].n_samples if self.members else 0

    @property
    def is_fitted(self) -> bool:
        return any(m.is_fitted for m in self.members)
