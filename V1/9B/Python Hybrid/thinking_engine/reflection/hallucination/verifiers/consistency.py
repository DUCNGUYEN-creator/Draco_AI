# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConsistencyVerifier
======================
Self-consistency check at the EVIDENCE level: if ``context`` carries
multiple independent reasoning paths (e.g. SelfConsistency's n_paths,
or MultiAgentDebate's per-expert opinions), checks whether the claim
agrees with the majority of them. A claim that only one reasoning path
out of several supports is a classic hallucination signature (the
well-known correlation between low self-consistency and factual
error), distinct from ContradictionVerifier (which checks against
retrieved EVIDENCE, not against the model's OWN other reasoning paths).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.enums import VerifierKind
from ..models.verification import VerificationResult


def _word_overlap_ratio(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class ConsistencyVerifier:
    name = "consistency"
    kind = VerifierKind.CONSISTENCY

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        reasoning_paths: List[str] = context.get("reasoning_paths", []) or []
        debate_opinions: Dict[int, str] = context.get("debate_opinions", {}) or {}

        candidates: List[str] = list(reasoning_paths) + list(debate_opinions.values())

        if not candidates:
            # No multi-path signal available — neutral, low-confidence pass-through.
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.6, confidence=0.2, issues=[],
            )
            return result.as_dict()

        agreements = [c for c in candidates if _word_overlap_ratio(claim, c) >= 0.15]
        agreement_ratio = len(agreements) / len(candidates)

        issues = []
        if agreement_ratio < 0.34:
            issues.append(
                f"Claim agrees with only {len(agreements)}/{len(candidates)} independent "
                f"reasoning paths — low self-consistency."
            )

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=agreement_ratio,
            confidence=min(0.3 + 0.1 * len(candidates), 0.9),
            issues=issues,
            metadata={"n_paths_compared": len(candidates), "n_agreeing": len(agreements)},
        )
        return result.as_dict()
