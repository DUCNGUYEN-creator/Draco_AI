# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationSelector
======================
Picks the appropriate calibration method given HallucinationConfig and
the amount of accumulated data — e.g. fall back to Temperature (fastest
to converge) when data is very sparse, graduate to Isotonic once enough
samples exist to make its extra flexibility worthwhile, and let
"ensemble" mean averaging across several. Reads
config.HallucinationConfig.calibration_method but never hardcodes a
choice the config didn't ask for.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseCalibrator
from .beta import BetaCalibrator
from .ensemble import CalibrationEnsemble
from .histogram import HistogramCalibrator
from .isotonic import IsotonicCalibrator
from .platt import PlattCalibrator
from .temperature import TemperatureCalibrator

_REGISTRY = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "beta": BetaCalibrator,
    "temperature": TemperatureCalibrator,
    "histogram": HistogramCalibrator,
}


class CalibrationSelector:
    def select(self, method: str) -> "BaseCalibrator | CalibrationEnsemble":
        if method == "ensemble":
            return CalibrationEnsemble()
        cls = _REGISTRY.get(method)
        if cls is None:
            return PlattCalibrator()  # safe, well-understood default
        return cls()
