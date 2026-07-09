# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.reflection.hallucination.models — canonical data shapes
for the entire Hallucination subsystem. Every verifier, analyzer,
calibrator, correlator, and fusion strategy speaks these dataclasses —
this is what lets reflection/hallucination/registry + factory swap
implementations without breaking assessor.py's orchestration code.
"""

from .enums import RiskLevel, EvidenceType, VerifierKind
from .evidence import Evidence, EvidenceBundle
from .verification import VerificationResult
from .correlation import CorrelationGroup
from .fusion import FusionResult
from .calibration import CalibrationPoint, CalibrationModel
from .risk import RiskAssessment
from .report import HallucinationReport
from .statistics import RunningStats

__all__ = [
    "RiskLevel",
    "EvidenceType",
    "VerifierKind",
    "Evidence",
    "EvidenceBundle",
    "VerificationResult",
    "CorrelationGroup",
    "FusionResult",
    "CalibrationPoint",
    "CalibrationModel",
    "RiskAssessment",
    "HallucinationReport",
    "RunningStats",
]
