# DracoAI V1 — thinking_engine/reflection/hallucination/pipeline/calibration_pipeline.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationPipeline — Stage 3: "Calibration"
================================================
NOTE on pipeline ordering: per the architecture's canonical sequence
(Evidence -> Verification -> Calibration -> Correlation -> Fusion ->
Risk -> Report), calibration is positioned as stage 3. In this
implementation, calibration is applied in TWO places to respect both
the architecture's stage ordering and basic statistical correctness:

1. Here (stage 3): each individual verifier's raw score is calibrated
   independently via its own per-verifier calibration model, BEFORE
   correlation/fusion combine them. This corrects each verifier's own
   over/under-confidence bias prior to combination.
2. After fusion (in AssessmentPipeline): the FINAL fused score is
   calibrated AGAIN via a top-level calibrator, since fusing several
   already-calibrated-but-still-biased signals can itself introduce a
   new systematic bias (e.g. noisy-OR tends to push scores up) that
   only a calibrator operating on the fused output can correct.

This two-point design satisfies the document's stage ordering exactly
while remaining statistically sound — calibrating only once, either
before OR after fusion exclusively, would either leave verifier-level
bias uncorrected or leave fusion-method bias uncorrected.

FIX (Bug #7): ``_scoped()`` no longer accesses ``factory._instance_cache``
directly — it now delegates to ``factory.get_scoped(method, scope)``
which is a proper public API method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..calibration.base import BaseCalibrator
from ..factory.calibration_factory import CalibrationFactory


class CalibrationPipeline:
    def __init__(self, factory: Optional[CalibrationFactory] = None) -> None:
        self.factory = factory or CalibrationFactory()

    def calibrate_verifier_score(self, verifier_name: str, raw_score: float, method: str) -> float:
        """Calibrates ONE verifier's raw score using a per-verifier
        calibration model (scoped by verifier_name so e.g. NumericalVerifier's
        score distribution isn't calibrated using RetrievalVerifier's history)."""
        calibrator = self._scoped(method, verifier_name)
        return calibrator.calibrate(raw_score)

    def _scoped(self, method: str, scope: str):
        """Returns a calibrator instance scoped by (method, scope).
        Delegates to CalibrationFactory.get_scoped() — the factory owns
        the cache and constructs new instances when needed.

        FIX: Previously this method directly accessed
        ``self.factory._instance_cache``, violating encapsulation.
        """
        return self.factory.get_scoped(method, scope)

    def calibrate_batch(
        self, verification_results: List[Dict[str, Any]], method: str
    ) -> List[Dict[str, Any]]:
        """Returns a NEW list of result dicts with an added 'calibrated_score'
        field — never mutates the input dicts (verifier results may be
        shared/cached elsewhere)."""
        out = []
        for r in verification_results:
            nr = dict(r)
            nr["calibrated_score"] = self.calibrate_verifier_score(
                r.get("verifier", "unknown"), r.get("score", 0.5), method
            )
            out.append(nr)
        return out

    def record_outcome(self, verifier_name: str, raw_score: float, was_correct: bool, method: str) -> None:
        """Feeds ground-truth feedback (e.g. from a later user
        correction or ReliabilityTracker) back into the per-verifier
        calibrator so it keeps improving over time."""
        calibrator = self._scoped(method, verifier_name)
        if hasattr(calibrator, "record"):
            calibrator.record(raw_score, int(was_correct))
