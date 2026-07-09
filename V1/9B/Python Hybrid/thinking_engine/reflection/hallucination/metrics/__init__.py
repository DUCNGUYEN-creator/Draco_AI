# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.metrics
====================================================
Quantitative quality metrics for the Hallucination subsystem ITSELF —
how well-calibrated are our calibrators, how reliable are our
verifiers, is verifier-score drift occurring over time. Consumed by
benchmarks/* and telemetry.py; never consumed by assessor.py's
per-request hot path (these are offline/monitoring metrics).
"""

from .confidence import mean_confidence, confidence_weighted_score
from .uncertainty import predictive_entropy
from .entropy import shannon_entropy, normalized_entropy
from .reliability import ReliabilityTracker
from .verifier_score import VerifierScoreTracker
from .calibration_error import expected_calibration_error, maximum_calibration_error
from .ece import compute_ece_bins
from .brier import brier_score
from .roc import roc_curve_points, auc_from_points
from .drift import DriftDetector
from .distribution import histogram, percentile

__all__ = [
    "mean_confidence",
    "confidence_weighted_score",
    "predictive_entropy",
    "shannon_entropy",
    "normalized_entropy",
    "ReliabilityTracker",
    "VerifierScoreTracker",
    "expected_calibration_error",
    "maximum_calibration_error",
    "compute_ece_bins",
    "brier_score",
    "roc_curve_points",
    "auc_from_points",
    "DriftDetector",
    "histogram",
    "percentile",
]
