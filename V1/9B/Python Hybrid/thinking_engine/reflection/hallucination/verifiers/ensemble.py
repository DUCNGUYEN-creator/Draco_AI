# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierEnsemble
===================
Runs a configured SET of verifiers over one claim and collects their
raw VerificationResult dicts. Deliberately does NOT fuse them into one
score — that is fusion/*.py's exclusive job (per the architecture:
"Assessor chỉ gọi verifier" / "Hallucination chỉ cung cấp Report").
This class is what registry.VerifierRegistry + factory.VerifierFactory
+ strategy.* configure to assemble the right verifier set per request.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.enums import VerifierKind


class VerifierEnsemble:
    name = "ensemble"
    kind = VerifierKind.ENSEMBLE

    def __init__(self, verifiers: List[Any]) -> None:
        """``verifiers`` is a list of verifier instances, each exposing
        ``.name`` and ``.verify(claim, evidence, context) -> dict``."""
        self.verifiers = verifiers

    def run_all(self, claim: str, evidence: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for v in self.verifiers:
            try:
                r = v.verify(claim, evidence, context)
            except Exception as exc:  # pragma: no cover — never let one bad verifier kill the request
                r = {
                    "verifier": getattr(v, "name", v.__class__.__name__),
                    "kind": "ensemble_error",
                    "claim": claim,
                    "score": 0.5,
                    "confidence": 0.0,
                    "issues": [f"Verifier raised an exception: {exc}"],
                    "metadata": {},
                }
            results.append(r)
        return results
