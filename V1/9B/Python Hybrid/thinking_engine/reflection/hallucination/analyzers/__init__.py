# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.analyzers
======================================================
Analyzers operate on the COLLECTION of VerificationResults a single
claim received (not on raw evidence like verifiers do) — they classify,
score severity, detect outliers, and check inter-verifier dependency
and coverage. Analyzers feed structured signals into calibration/
correlation/fusion; they never re-derive a score from scratch.
"""

from .taxonomy import HallucinationTaxonomy, HallucinationType
from .severity import SeverityAnalyzer
from .verifier import VerifierAgreementAnalyzer
from .contradiction import ContradictionAnalyzer
from .consistency import ConsistencyAnalyzer
from .uncertainty import UncertaintyAnalyzer
from .dependency import DependencyAnalyzer
from .coverage import CoverageAnalyzer
from .outlier import OutlierAnalyzer
from .statistics import StatisticsAnalyzer

__all__ = [
    "HallucinationTaxonomy",
    "HallucinationType",
    "SeverityAnalyzer",
    "VerifierAgreementAnalyzer",
    "ContradictionAnalyzer",
    "ConsistencyAnalyzer",
    "UncertaintyAnalyzer",
    "DependencyAnalyzer",
    "CoverageAnalyzer",
    "OutlierAnalyzer",
    "StatisticsAnalyzer",
]
