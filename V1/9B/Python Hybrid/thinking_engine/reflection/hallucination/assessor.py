# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Assessor — public entry point for reflection.hallucination
============================================================
``Assessor`` is the ONLY class from ``reflection.hallucination`` that
anything outside this package is supposed to instantiate. Every other
class in this package is internal implementation detail; only
``HallucinationReport`` (from ``models.report``) crosses the boundary
as a *result* object.

Usage
-----
    from thinking_engine.reflection.hallucination import Assessor
    assessor = Assessor(config=engine.config.hallucination)
    report   = assessor.assess(answer=answer_text, context=thinking_state_dict)
    report_dict = report.as_dict()   # JSON-safe for logging / PromptCompiler

Architecture note
-----------------
Assessor orchestrates the pipeline and telemetry; it contains zero
verification logic itself. "Assessor chỉ orchestrate" — per the
architecture document's own wording.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ...config import HallucinationConfig
from .cache.evidence_cache import EvidenceCache
from .cache.verifier_cache import VerifierCache
from .factory.calibration_factory import CalibrationFactory
from .factory.fusion_factory import FusionFactory
from .factory.verifier_factory import VerifierFactory
from .models.report import HallucinationReport
from .pipeline.assessment_pipeline import AssessmentPipeline
from .telemetry import get_telemetry


class Assessor:
    """Thread-safe; safe to share across concurrent request-handling threads
    (each ``assess()`` call is independently stateless at this level —
    statefulness lives in the calibration-model instances inside
    CalibrationFactory, which are themselves lock-protected through
    Python's GIL for the micro-increment updates they perform)."""

    def __init__(self, config: Optional[HallucinationConfig] = None) -> None:
        self._config = config or HallucinationConfig()
        self._telemetry = get_telemetry()

        # Shared, cached factories — constructed once per Assessor instance.
        self._verifier_factory = VerifierFactory()
        self._fusion_factory = FusionFactory()
        self._calibration_factory = CalibrationFactory()
        self._evidence_cache = EvidenceCache(
            max_size=self._config.cache_size,
            ttl_seconds=self._config.cache_ttl_seconds,
        )
        self._verifier_cache = VerifierCache(
            max_size=self._config.cache_size * 2,
            ttl_seconds=self._config.cache_ttl_seconds,
        )

        self._pipeline = AssessmentPipeline(
            verifier_factory=self._verifier_factory,
            fusion_factory=self._fusion_factory,
            calibration_factory=self._calibration_factory,
            evidence_cache=self._evidence_cache,
            verifier_cache=self._verifier_cache,
        )

    def assess(
        self,
        answer: str,
        context: Dict[str, Any],
        strategy_name: Optional[str] = None,
        max_claims: int = 10,
    ) -> HallucinationReport:
        """Run the full 6-stage Hallucination pipeline over ``answer``.

        Parameters
        ----------
        answer        : the model's generated answer text
        context       : ThinkingState-derived context dict containing any
                        of: reasoning_path, rag_docs, memory_summary,
                        tool_results, subgoals, debate_opinions,
                        known_citation_ids, reasoning_paths
        strategy_name : overrides config.strategy for this call only
        max_claims    : max checkable claims extracted from answer
        """
        started = time.time()
        strategy = strategy_name or self._config.strategy

        report = self._pipeline.assess(
            answer=answer,
            context=context,
            strategy_name=strategy,
            risk_thresholds=self._config.risk_thresholds,
            max_claims=max_claims,
        )

        latency_ms = (time.time() - started) * 1000.0
        self._telemetry.record_request(latency_ms, report.risk_level.value)
        self._telemetry.record_fusion(
            getattr(self._config, "fusion_method", "noisy_or")
        )

        report.latency_ms = latency_ms
        return report

    def record_outcome(
        self,
        claim: str,
        raw_verifier_scores: Dict[str, float],
        was_correct: bool,
    ) -> None:
        """Feed ground-truth feedback (from user correction / eval harness)
        back into the per-verifier calibration models so they keep
        improving. ``raw_verifier_scores`` is {verifier_name: raw_score}.
        """
        method = self._config.calibration_method
        cal_pipeline = self._pipeline._cal_pipe
        for name, score in raw_verifier_scores.items():
            cal_pipeline.record_outcome(name, score, was_correct, method)
        self._telemetry.record_calibration()

    def telemetry_snapshot(self) -> Dict[str, Any]:
        return self._telemetry.snapshot()
