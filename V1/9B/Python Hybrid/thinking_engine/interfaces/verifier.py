# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.interfaces.verifier
=====================================
The contract every Hallucination verifier must satisfy. This is the
single most important interface in the Verification layer: it is what
lets reflection/hallucination/registry + factory plug in new verifiers
without touching assessor.py.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class Verifier(Protocol):
    """A Verifier inspects one (claim, evidence, context) triple and
    returns a VerificationResult-shaped dict (see
    reflection/hallucination/models/verification.py for the canonical
    dataclass). Verifiers NEVER mutate global state and NEVER call the
    LLM to "fix" anything — they only judge.
    """

    name: str

    def verify(
        self,
        claim: str,
        evidence: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a dict with at least:
        {
            "verifier": str,
            "score": float,        # 0.0 (unsupported) .. 1.0 (fully supported)
            "confidence": float,   # how much to trust this verifier's score
            "issues": list[str],
        }
        """
        ...
