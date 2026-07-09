# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationBenchmark
======================
Evaluates calibration quality (ECE + MCE) for each method on a labelled
sample after fitting. Confirms calibrators actually reduce calibration
error vs. raw (uncalibrated) scores.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..metrics.calibration_error import expected_calibration_error, maximum_calibration_error
from ..registry.calibration_registry import CalibrationRegistry

_SAMPLE: List[Tuple[float, int]] = [
    (0.9, 1), (0.85, 1), (0.8, 1), (0.75, 1), (0.7, 1),
    (0.3, 0), (0.25, 0), (0.2, 0), (0.15, 0), (0.1, 0),
]


class CalibrationBenchmark:
    def __init__(self, registry: CalibrationRegistry | None = None) -> None:
        self.registry = registry or CalibrationRegistry()

    def run(self, sample: List[Tuple[float, int]] | None = None) -> Dict[str, Dict]:
        sample = sample or _SAMPLE
        results = {}
        for name in self.registry.available():
            cal = self.registry.create(name)
            for raw, label in sample:
                cal.record(raw, label)
            preds = [cal.calibrate(raw) for raw, _ in sample]
            labels = [label for _, label in sample]
            raw_scores = [raw for raw, _ in sample]
            results[name] = {
                "ece_calibrated": round(expected_calibration_error(preds, labels), 4),
                "ece_raw": round(expected_calibration_error(raw_scores, labels), 4),
                "mce_calibrated": round(maximum_calibration_error(preds, labels), 4),
            }
        return results
