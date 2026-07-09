# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""HallucinationReport — the final, top-level output of the entire
Hallucination subsystem; this is the ONLY object reflection/* (outside
of hallucination/) is allowed to depend on. Everything else inside
reflection/hallucination/ is private implementation detail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .enums import RiskLevel
from .risk import RiskAssessment


@dataclass
class HallucinationReport:
    risk_score: float
    risk_level: RiskLevel
    per_claim: List[RiskAssessment] = field(default_factory=list)
    top_issues: List[str] = field(default_factory=list)
    strategy_used: str = "balanced"
    n_verifiers_run: int = 0
    n_claims_checked: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value if hasattr(self.risk_level, "value") else str(self.risk_level),
            "top_issues": list(self.top_issues),
            "strategy_used": self.strategy_used,
            "n_verifiers_run": self.n_verifiers_run,
            "n_claims_checked": self.n_claims_checked,
            "latency_ms": round(self.latency_ms, 3),
            "per_claim": [
                {
                    "claim": ra.claim,
                    "risk_score": round(ra.risk_score, 4),
                    "risk_level": ra.risk_level.value if hasattr(ra.risk_level, "value") else str(ra.risk_level),
                    "issues": list(ra.contributing_issues),
                }
                for ra in self.per_claim
            ],
        }
