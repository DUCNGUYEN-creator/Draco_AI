# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination
==========================================
Deep Hallucination Detection subsystem — the "Verification" layer of
the Architecture's Infrastructure / Cognition / Verification split.

Public surface (the ONLY objects reflection/* outside this package
should ever import):
    Assessor            — entry point, feeds an answer through the
                          6-stage pipeline, returns a HallucinationReport
    HallucinationReport — the final output dataclass; serializable to
                          dict/JSON via .as_dict()

Every other class in this package is internal implementation. Import
them directly for testing, benchmarking, or plugin-registration; never
import them from the engine's main pipeline code.
"""

from .assessor import Assessor
from .models.report import HallucinationReport
from .models.enums import RiskLevel, EvidenceType, VerifierKind
from .telemetry import get_telemetry
from .version import HALLUCINATION_VERSION

__all__ = [
    "Assessor",
    "HallucinationReport",
    "RiskLevel",
    "EvidenceType",
    "VerifierKind",
    "get_telemetry",
    "HALLUCINATION_VERSION",
]
