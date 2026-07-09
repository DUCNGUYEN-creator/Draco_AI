# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ReasoningVerifier
====================
Checks that a claim is actually traceable to the reasoning that
allegedly produced it — best_branch, debate_synthesis, the KG
reasoning_path. Distinct from ConsistencyVerifier (agreement ACROSS
multiple independent paths) and reasoning.thinking.chain_verifier.
ChainOfThoughtVerifier (soundness of the THOUGHTS themselves, before
any final claim exists): ReasoningVerifier specifically asks "does
this claim follow from the SELECTED reasoning trace", i.e. detects
non-sequiturs where the final answer asserts something the chosen
reasoning chain never actually established.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.enums import VerifierKind
from ..models.verification import VerificationResult


class ReasoningVerifier:
    name = "reasoning"
    kind = VerifierKind.REASONING

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        best_branch: str = context.get("best_branch", "") or ""
        reasoning_path: List[str] = context.get("reasoning_path", []) or []

        trace_text = " ".join([best_branch] + reasoning_path).lower()
        if not trace_text.strip():
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.5, confidence=0.1, issues=[],
            )
            return result.as_dict()

        claim_words = {w for w in claim.lower().split() if len(w) > 2}
        trace_words = {w for w in trace_text.split() if len(w) > 2}
        if not claim_words:
            score = 0.5
        else:
            score = len(claim_words & trace_words) / len(claim_words)

        issues: List[str] = []
        if score < 0.1:
            issues.append(
                "Claim shares almost no content with the selected reasoning trace — "
                "possible non-sequitur (conclusion doesn't follow from the reasoning shown)."
            )

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=min(score, 1.0),
            confidence=0.5,
            issues=issues,
        )
        return result.as_dict()
