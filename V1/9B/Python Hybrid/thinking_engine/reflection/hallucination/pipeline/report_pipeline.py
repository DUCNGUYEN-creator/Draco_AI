# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReportPipeline — Stage "Report" (after Risk)
================================================
Assembles the final HallucinationReport from a list of per-claim
RiskAssessments — the LAST stage in the canonical pipeline. This is the
ONLY object the rest of reflection/* (outside hallucination/) and the
Output stage are allowed to depend on.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from ..models.enums import RiskLevel
from ..models.report import HallucinationReport
from ..models.risk import RiskAssessment


class ReportPipeline:
    def assemble(
        self,
        per_claim: List[RiskAssessment],
        strategy_used: str,
        n_verifiers_run: int,
        started_at: float,
    ) -> HallucinationReport:
        if not per_claim:
            return HallucinationReport(
                risk_score=0.0,
                risk_level=RiskLevel.NONE,
                strategy_used=strategy_used,
                n_verifiers_run=n_verifiers_run,
                n_claims_checked=0,
                latency_ms=(time.time() - started_at) * 1000.0,
            )

        # Overall risk = the WORST (highest) per-claim risk, not the
        # average — one severely hallucinated claim in an otherwise-good
        # answer should not be diluted away by several fine claims.
        worst = max(per_claim, key=lambda ra: ra.risk_score)

        top_issues: List[str] = []
        seen = set()
        for ra in sorted(per_claim, key=lambda r: r.risk_score, reverse=True):
            for issue in ra.contributing_issues:
                if issue not in seen:
                    seen.add(issue)
                    top_issues.append(issue)
            if len(top_issues) >= 8:
                break

        return HallucinationReport(
            risk_score=worst.risk_score,
            risk_level=worst.risk_level,
            per_claim=per_claim,
            top_issues=top_issues[:8],
            strategy_used=strategy_used,
            n_verifiers_run=n_verifiers_run,
            n_claims_checked=len(per_claim),
            latency_ms=(time.time() - started_at) * 1000.0,
        )
