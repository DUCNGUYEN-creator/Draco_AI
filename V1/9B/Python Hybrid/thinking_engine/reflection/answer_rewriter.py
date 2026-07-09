# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
AnswerRewriter
================
The final Verification-layer gate before Output. Reads the
Hallucination report (risk_level/risk_score), the ethical-filter note,
and the consistency-check issues, and decides whether/how the answer
needs a rewrite instruction appended — formalizing the
``post_generation_check`` + ``EthicalFilter.build_rewrite_instruction``
combination that engine_v1.py applied ad-hoc at the end of process().
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..constants import RISK_CRITICAL, RISK_HIGH


class AnswerRewriter:
    def needs_rewrite(self, hallucination_report: Optional[Dict[str, Any]], is_ethical: bool) -> bool:
        if not is_ethical:
            return True
        if hallucination_report and hallucination_report.get("risk_level") in (RISK_HIGH, RISK_CRITICAL):
            return True
        return False

    def build_rewrite_instruction(
        self,
        hallucination_report: Optional[Dict[str, Any]],
        ethical_note: str,
    ) -> str:
        parts = []
        if ethical_note:
            parts.append(ethical_note)
        if hallucination_report and hallucination_report.get("risk_level") in (RISK_HIGH, RISK_CRITICAL):
            top_issues = hallucination_report.get("top_issues", [])[:3]
            issues_text = "; ".join(top_issues) if top_issues else "unverified claims detected"
            parts.append(
                f"[HALLUCINATION RISK: {hallucination_report.get('risk_level')}] "
                f"Revise the answer to address: {issues_text}. "
                f"Only state what is well-supported; mark uncertain claims explicitly."
            )
        return "\n".join(parts)

    def apply_uncertainty_tags(self, answer: str, tagged_answer: str) -> str:
        """Prefer the uncertainty-tagged version when it actually added
        tags (i.e. differs from the original); otherwise keep the plain
        answer to avoid noisy [confidence:X] tags on already-clear text."""
        return tagged_answer if tagged_answer != answer else answer
