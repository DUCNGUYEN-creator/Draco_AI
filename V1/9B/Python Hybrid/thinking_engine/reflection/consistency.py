# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConsistencyChecker
=====================
Reflection-layer self-consistency check: compares a final answer
against the reasoning trace that led to it (best_branch, debate
synthesis, sc_path) for gross mismatches — distinct from
reasoning/thinking/chain_verifier.py (which checks the THOUGHTS for
internal soundness before any answer exists) and distinct from
reflection/hallucination/verifiers/consistency.py (which is one
evidence-scoring verifier inside the hallucination ensemble).
"""

from __future__ import annotations

from typing import Any, Dict, List


class ConsistencyChecker:
    def check(self, answer: str, thought_plan: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        answer_words = set(answer.lower().split())

        best_branch = thought_plan.get("best_branch", "")
        if best_branch:
            branch_words = set(best_branch.lower().split())
            overlap = len(answer_words & branch_words) / max(len(branch_words), 1)
            if overlap < 0.05:
                issues.append(
                    "Final answer shares almost no vocabulary with the selected "
                    "reasoning branch — possible drift from the plan."
                )

        debate_synthesis = thought_plan.get("debate_synthesis", "")
        if debate_synthesis:
            debate_words = set(debate_synthesis.lower().split())
            overlap = len(answer_words & debate_words) / max(len(debate_words), 1)
            if overlap < 0.03:
                issues.append("Final answer diverges strongly from the debate synthesis.")

        return {
            "issues": issues,
            "consistent": len(issues) == 0,
        }
