# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SymbolicVerifier
===================
For claims phrased as formal propositional-logic statements ("if A and
B then C", "A or not A is always true"), uses
reasoning.cognitive.symbolic_reasoning.SymbolicReasoner to check
tautology/contradiction/satisfiability exactly, instead of the
ContradictionVerifier's fuzzy antonym-pair heuristic. Complementary,
not redundant: SymbolicVerifier only fires on claims it can parse as
pure boolean expressions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ....reasoning.cognitive.symbolic_reasoning import SymbolicReasoner
from ..models.enums import VerifierKind
from ..models.verification import VerificationResult

_LOGIC_HINT_RE = re.compile(r"\b(and|or|not)\b", re.IGNORECASE)


class SymbolicVerifier:
    name = "symbolic"
    kind = VerifierKind.SYMBOLIC

    def __init__(self) -> None:
        self._reasoner = SymbolicReasoner()

    def _looks_symbolic(self, claim: str) -> bool:
        return bool(_LOGIC_HINT_RE.search(claim)) and len(self._reasoner.extract_variables(claim)) >= 1

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self._looks_symbolic(claim):
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.5, confidence=0.0, issues=[],
            )
            return result.as_dict()

        issues: List[str] = []
        score = 0.6
        confidence = 0.3
        try:
            variables = self._reasoner.extract_variables(claim)
            if 0 < len(variables) <= 6:  # cap truth-table size for safety/perf
                claims_tautology = re.search(r"\balways\s+true\b|luôn\s+đúng", claim, re.IGNORECASE)
                claims_contradiction = re.search(r"\balways\s+false\b|luôn\s+sai", claim, re.IGNORECASE)
                if claims_tautology:
                    is_tauto = self._reasoner.is_tautology(claim)
                    score = 1.0 if is_tauto else 0.1
                    confidence = 0.85
                    if not is_tauto:
                        issues.append("Claimed tautology does not hold for all variable assignments.")
                elif claims_contradiction:
                    is_contra = self._reasoner.is_contradiction(claim)
                    score = 1.0 if is_contra else 0.1
                    confidence = 0.85
                    if not is_contra:
                        issues.append("Claimed contradiction is not false under all assignments.")
        except Exception as exc:  # pragma: no cover — defensive, never raise into the ensemble
            issues.append(f"Symbolic evaluation failed: {exc}")
            score, confidence = 0.5, 0.1

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=confidence,
            issues=issues,
        )
        return result.as_dict()
