# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
AssessmentPipeline
=====================
The top-level orchestrator that connects every stage of the canonical
Hallucination sub-pipeline:

    Evidence -> Verification -> Calibration -> Correlation -> Fusion -> Risk -> Report

assessor.py is the ONLY public entry point into this pipeline — it
handles the claim-extraction step (splitting an answer into checkable
claims) and feeds each claim through this pipeline in order.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..cache.evidence_cache import EvidenceCache
from ..cache.verifier_cache import VerifierCache
from ..factory.calibration_factory import CalibrationFactory
from ..factory.fusion_factory import FusionFactory
from ..factory.verifier_factory import VerifierFactory
from ..models.enums import RiskLevel
from ..models.report import HallucinationReport
from ..models.risk import RiskAssessment
from ..strategy import get_strategy
from .calibration_pipeline import CalibrationPipeline
from .correlation_pipeline import CorrelationPipeline
from .evidence_pipeline import EvidencePipeline
from .fusion_pipeline import FusionPipeline
from .report_pipeline import ReportPipeline
from .verification_pipeline import VerificationPipeline

# Risk-level decision thresholds — identical to config.HallucinationConfig defaults
_DEFAULT_THRESHOLDS = {"low": 0.15, "medium": 0.35, "high": 0.60, "critical": 0.85}


def _split_claims(text: str, max_claims: int = 10) -> List[str]:
    """Splits an answer text into individual checkable claims (sentences
    longer than ~10 chars). Simple regex split — not trying to do full
    NLP sentence-boundary detection here; edge-case mis-splits are
    acceptable since the verifiers are designed to be robust to imperfect
    claim boundaries.
    """
    raw = re.split(r"(?<=[.!?؟।])\s+", text.strip())
    claims = [s.strip() for s in raw if len(s.strip()) > 10]
    return claims[:max_claims]


class AssessmentPipeline:
    def __init__(
        self,
        verifier_factory: Optional[VerifierFactory] = None,
        fusion_factory: Optional[FusionFactory] = None,
        calibration_factory: Optional[CalibrationFactory] = None,
        evidence_cache: Optional[EvidenceCache] = None,
        verifier_cache: Optional[VerifierCache] = None,
    ) -> None:
        self._ev_pipe = EvidencePipeline(cache=evidence_cache)
        self._ver_pipe = VerificationPipeline(
            factory=verifier_factory,
            cache=verifier_cache,
        )
        self._cal_pipe = CalibrationPipeline(factory=calibration_factory)
        self._cor_pipe = CorrelationPipeline()
        self._fus_pipe = FusionPipeline(factory=fusion_factory)
        self._rep_pipe = ReportPipeline()

    def assess(
        self,
        answer: str,
        context: Dict[str, Any],
        strategy_name: str = "balanced",
        risk_thresholds: Optional[Dict[str, float]] = None,
        max_claims: int = 10,
    ) -> HallucinationReport:
        """Run the full 6-stage pipeline over every extracted claim in
        `answer`. Returns one HallucinationReport whose risk_score/
        risk_level reflects the WORST individual claim (not the average).

        Parameters
        ----------
        answer          : str — the model's generated answer text
        context         : dict — shared context supplied by ThinkingState
                          (reasoning_path, rag_docs, memory_summary,
                          tool_results, subgoals, debate_opinions, etc.)
        strategy_name   : "fast" | "balanced" | "paranoid" | "custom"
        risk_thresholds : optional override for RiskLevel bucket edges
        max_claims      : hard cap on claims checked per call
        """
        started_at = time.time()
        thresholds = risk_thresholds or _DEFAULT_THRESHOLDS
        strat = get_strategy(strategy_name)
        verifier_names = strat.verifier_names
        fusion_method = strat.fusion_method
        calibration_method = strat.calibration_method

        claims = _split_claims(answer, max_claims)
        if not claims:
            claims = [answer[:200]] if answer.strip() else []

        per_claim_assessments: List[RiskAssessment] = []

        for claim in claims:
            # ── Stage 1: Evidence ────────────────────────────────────
            bundle = self._ev_pipe.gather(claim, context)

            # ── Stage 2: Verification ─────────────────────────────────
            raw_results = self._ver_pipe.run(claim, bundle, context, verifier_names)

            # ── Stage 3: Calibration (per-verifier) ──────────────────
            calibrated_results = self._cal_pipe.calibrate_batch(raw_results, calibration_method)

            # ── Stage 4: Correlation (decorrelation → weights) ───────
            signals = self._cor_pipe.to_fusion_signals(calibrated_results)

            if not signals:
                per_claim_assessments.append(
                    RiskAssessment(
                        claim=claim,
                        risk_score=0.0,
                        risk_level=RiskLevel.NONE,
                        contributing_issues=[],
                        calibrated=False,
                    )
                )
                continue

            # ── Stage 5: Fusion ───────────────────────────────────────
            fusion_result = self._fus_pipe.fuse(signals, fusion_method)

            # ── Post-fusion calibration (corrects fusion-method bias) ─
            fused_raw = fusion_result.fused_score
            top_level_cal = self._cal_pipe._scoped(calibration_method, "_fused")
            fused_calibrated = top_level_cal.calibrate(fused_raw)

            # ── Stage 6a: Risk ────────────────────────────────────────
            risk_level = RiskLevel.from_score(fused_calibrated, thresholds)
            issues = []
            for r in calibrated_results:
                issues.extend(r.get("issues", []))

            per_claim_assessments.append(
                RiskAssessment(
                    claim=claim,
                    risk_score=round(fused_calibrated, 4),
                    risk_level=risk_level,
                    contributing_issues=issues[:5],
                    calibrated=True,
                )
            )

        # ── Stage 6b: Report ──────────────────────────────────────────
        return self._rep_pipe.assemble(
            per_claim=per_claim_assessments,
            strategy_used=strategy_name,
            n_verifiers_run=len(verifier_names),
            started_at=started_at,
        )
