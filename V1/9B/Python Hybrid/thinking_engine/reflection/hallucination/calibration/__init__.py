# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.calibration
========================================================
Maps a raw fused verifier score into a CALIBRATED probability that
matches empirical frequency (i.e. of claims scored 0.7, roughly 70%
should actually turn out correct). This is the full method zoo — far
beyond reflection.confidence_calibrator.ConfidenceCalibrator's always-
on, lightweight Platt-only calibrator used directly by the pipeline.
"""

from .base import BaseCalibrator
from .platt import PlattCalibrator
from .isotonic import IsotonicCalibrator
from .beta import BetaCalibrator
from .temperature import TemperatureCalibrator
from .histogram import HistogramCalibrator
from .ensemble import CalibrationEnsemble
from .selector import CalibrationSelector

__all__ = [
    "BaseCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "BetaCalibrator",
    "TemperatureCalibrator",
    "HistogramCalibrator",
    "CalibrationEnsemble",
    "CalibrationSelector",
]
