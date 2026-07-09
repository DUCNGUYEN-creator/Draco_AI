# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""VerificationResult — the canonical return shape every Verifier produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .enums import VerifierKind


@dataclass
class VerificationResult:
    verifier: str
    kind: VerifierKind
    claim: str
    score: float  # 0.0 (unsupported) .. 1.0 (fully supported)
    confidence: float  # how much to trust THIS verifier's score, 0..1
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def failure_probability(self) -> float:
        """Probability this claim is a hallucination, AS JUDGED BY THIS
        VERIFIER ALONE — fusion.py combines many of these.
        failure_prob = (1 - score) weighted by this verifier's confidence;
        an unconfident verifier's opinion is pulled toward 0.5 (uninformative)."""
        raw = 1.0 - self.score
        return raw * self.confidence + 0.5 * (1.0 - self.confidence)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verifier": self.verifier,
            "kind": self.kind.value if hasattr(self.kind, "value") else str(self.kind),
            "claim": self.claim,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "issues": list(self.issues),
            "metadata": dict(self.metadata),
        }
