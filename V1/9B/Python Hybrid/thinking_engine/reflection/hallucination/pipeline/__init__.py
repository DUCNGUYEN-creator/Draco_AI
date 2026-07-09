# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.pipeline
=====================================================
The canonical 6-stage Hallucination sub-pipeline, exactly as specified
in the architecture document:

    Evidence -> Verification -> Calibration -> Correlation -> Fusion -> Risk -> Report

Each stage is its own module so any one of them can be swapped,
profiled, or unit-tested independently. assessment_pipeline.py is the
top-level orchestrator assessor.py actually calls.
"""

from .evidence_pipeline import EvidencePipeline
from .verification_pipeline import VerificationPipeline
from .calibration_pipeline import CalibrationPipeline
from .correlation_pipeline import CorrelationPipeline
from .fusion_pipeline import FusionPipeline
from .report_pipeline import ReportPipeline
from .assessment_pipeline import AssessmentPipeline

__all__ = [
    "EvidencePipeline",
    "VerificationPipeline",
    "CalibrationPipeline",
    "CorrelationPipeline",
    "FusionPipeline",
    "ReportPipeline",
    "AssessmentPipeline",
]
