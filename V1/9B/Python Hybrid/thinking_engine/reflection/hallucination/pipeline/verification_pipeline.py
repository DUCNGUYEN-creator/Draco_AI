# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerificationPipeline — Stage 2: "Verification"
==================================================
Runs the configured verifier ensemble over a claim+evidence pair,
checking VerifierCache first. Produces the raw List[VerificationResult
dict] that every subsequent stage (Calibration, Correlation, Fusion)
consumes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ....utils.hashing import stable_hash
from ...hallucination.factory.verifier_factory import VerifierFactory
from ..cache.verifier_cache import VerifierCache
from ..models.evidence import EvidenceBundle


class VerificationPipeline:
    def __init__(
        self,
        factory: Optional[VerifierFactory] = None,
        cache: Optional[VerifierCache] = None,
    ) -> None:
        self.factory = factory or VerifierFactory()
        self.cache = cache or VerifierCache()

    def _evidence_signature(self, bundle: EvidenceBundle) -> str:
        return stable_hash(*sorted(e.text for e in bundle.items)) if bundle.items else "empty"

    def run(
        self,
        claim: str,
        evidence: EvidenceBundle,
        context: Dict[str, Any],
        verifier_names: List[str],
    ) -> List[Dict[str, Any]]:
        ev_sig = self._evidence_signature(evidence)
        results: List[Dict[str, Any]] = []

        for name in verifier_names:
            cached = self.cache.get(name, claim, ev_sig)
            if cached is not None:
                results.append(cached)
                continue
            verifier = self.factory.get(name)
            try:
                r = verifier.verify(claim, evidence, context)
            except Exception as exc:  # pragma: no cover — defensive, mirrors VerifierEnsemble's guard
                r = {
                    "verifier": name,
                    "kind": "error",
                    "claim": claim,
                    "score": 0.5,
                    "confidence": 0.0,
                    "issues": [f"Verifier '{name}' raised an exception: {exc}"],
                    "metadata": {},
                }
            self.cache.set(name, claim, ev_sig, r)
            results.append(r)

        return results
