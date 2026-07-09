# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
NumericalVerifier
====================
For claims containing numbers/arithmetic, re-derives the result using
tools.sandbox.SafeASTEvaluator wherever the claim contains a checkable
expression, and compares against the asserted value — catching the
"confidently wrong arithmetic" hallucination pattern that pure
lexical-overlap verifiers (RetrievalVerifier) cannot detect at all
(an evidence passage can lexically "support" a wrong number if the
right number also appears nearby).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ....tools.sandbox import SafeASTEvaluator
from ..models.enums import VerifierKind
from ..models.verification import VerificationResult

# "<expr> = <result>" or "<expr> bằng <result>"
_EQUATION_RE = re.compile(
    r"([\d\.\s\+\-\*\/\(\)%]{3,40})\s*(?:=|bằng|equals)\s*(-?\d+(?:\.\d+)?)"
)


class NumericalVerifier:
    name = "numerical"
    kind = VerifierKind.NUMERICAL

    def __init__(self) -> None:
        self._evaluator = SafeASTEvaluator()

    def _extract_equations(self, claim: str) -> List[tuple]:
        out = []
        for m in _EQUATION_RE.finditer(claim):
            expr, asserted = m.group(1).strip(), m.group(2)
            if any(op in expr for op in "+-*/"):
                out.append((expr, asserted))
        return out

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        equations = self._extract_equations(claim)
        if not equations:
            # No checkable arithmetic in this claim — verifier abstains
            # (uninformative result, not a vote of confidence).
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.5, confidence=0.05, issues=[],
            )
            return result.as_dict()

        issues: List[str] = []
        n_correct = 0
        for expr, asserted in equations:
            computed = self._evaluator.evaluate(expr)
            if computed.startswith("Error"):
                continue
            try:
                if abs(float(computed) - float(asserted)) < 1e-6:
                    n_correct += 1
                else:
                    issues.append(f"'{expr.strip()} = {asserted}' but actual result is {computed}")
            except ValueError:
                continue

        n_checked = len(equations)
        score = n_correct / n_checked if n_checked else 0.5
        confidence = 0.9 if n_checked else 0.05  # very confident when arithmetic is checkable

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=confidence,
            issues=issues,
            metadata={"n_equations_checked": n_checked, "n_correct": n_correct},
        )
        return result.as_dict()
