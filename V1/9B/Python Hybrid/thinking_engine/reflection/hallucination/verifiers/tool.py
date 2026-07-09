# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ToolVerifier
==============
Checks a claim against actually-executed tool results
(tools.executor.ToolExecutor output) — the highest-trust evidence
source available (SourceManager assigns SOURCE_TOOL = 0.95 trust by
default), since a tool result is empirically derived, not generated.
Flags claims that state a different value than what the tool actually
returned, e.g. the model says "the result is 42" right after a
calculator tool call returned 41.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..models.enums import VerifierKind
from ..models.verification import VerificationResult


class ToolVerifier:
    name = "tool"
    kind = VerifierKind.TOOL

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        tool_results: List[Dict[str, Any]] = context.get("tool_results", []) or []
        successful = [r for r in tool_results if r.get("ok")]

        if not successful:
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.6, confidence=0.0, issues=[],
            )
            return result.as_dict()

        issues: List[str] = []
        n_matched = 0
        for r in successful:
            output = str(r.get("output", "")).strip()
            if not output:
                continue
            if output in claim:
                n_matched += 1
            else:
                # Try numeric-tolerant comparison for calculator-style outputs.
                try:
                    claim_numbers = re.findall(r"-?\d+(?:\.\d+)?", claim)
                    if any(abs(float(n) - float(output)) < 1e-6 for n in claim_numbers):
                        n_matched += 1
                    else:
                        issues.append(
                            f"Tool '{r.get('tool')}' returned '{output}' but claim doesn't reflect it."
                        )
                except ValueError:
                    pass

        score = n_matched / len(successful) if successful else 0.5
        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=0.85,  # tool output is high-trust ground truth
            issues=issues,
            metadata={"n_tool_results": len(successful), "n_matched": n_matched},
        )
        return result.as_dict()
