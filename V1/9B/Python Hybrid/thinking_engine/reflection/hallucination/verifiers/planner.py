# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PlannerVerifier
==================
Checks a claim against the Planning-stage's own subgoals/goal
decomposition for "plan abandonment" hallucinations — e.g. the final
answer asserts something that contradicts or skips a sub-goal the
engine itself committed to earlier in the same turn. Distinct from
reflection.consistency.ConsistencyChecker (which compares the WHOLE
final answer to the best reasoning branch) — this verifier operates
per-claim, against the structured plan specifically.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..models.enums import VerifierKind
from ..models.verification import VerificationResult


class PlannerVerifier:
    name = "planner"
    kind = VerifierKind.PLANNER

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        subgoals: List[str] = context.get("subgoals", []) or []
        goal_decomposition: List[str] = context.get("goal_decomposition", []) or []
        plan_items = subgoals + goal_decomposition

        if not plan_items:
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.6, confidence=0.0, issues=[],
            )
            return result.as_dict()

        claim_words = set(claim.lower().split())
        coverage = 0
        for item in plan_items:
            item_words = set(item.lower().split())
            if claim_words & item_words:
                coverage += 1

        ratio = coverage / len(plan_items)
        issues: List[str] = []
        if ratio == 0:
            issues.append("Claim does not reference any planned sub-goal or decomposition step.")

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=min(0.4 + ratio, 1.0),  # baseline 0.4 so absence of plan-overlap isn't fatal alone
            confidence=0.4,
            issues=issues,
            metadata={"n_plan_items": len(plan_items), "n_overlapping": coverage},
        )
        return result.as_dict()
