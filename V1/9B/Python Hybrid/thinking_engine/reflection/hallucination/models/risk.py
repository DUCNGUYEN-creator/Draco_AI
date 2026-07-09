# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""RiskAssessment — the per-claim risk verdict, before being rolled up
into the full HallucinationReport (which covers potentially many
claims extracted from one answer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .enums import RiskLevel


@dataclass
class RiskAssessment:
    claim: str
    risk_score: float
    risk_level: RiskLevel
    contributing_issues: List[str] = field(default_factory=list)
    calibrated: bool = False
