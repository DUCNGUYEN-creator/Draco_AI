# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CitationVerifier
===================
Checks that every citation marker referenced by a claim
(``[citation_id]`` style, as produced by
knowledge.citation.CitationTracker) actually maps back to a real
registered document — catches the "fabricated citation" hallucination
pattern, where a model invents a plausible-looking [abc123] tag that
was never actually retrieved.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from ..models.enums import VerifierKind
from ..models.verification import VerificationResult

_CITATION_RE = re.compile(r"\[([a-f0-9]{6,12})\]")


class CitationVerifier:
    name = "citation"
    kind = VerifierKind.CITATION

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        cited_ids = _CITATION_RE.findall(claim)
        if not cited_ids:
            # No citations made — neither praiseworthy nor suspicious on its own.
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.6, confidence=0.0, issues=[],
            )
            return result.as_dict()

        known_ids = set(context.get("known_citation_ids", []) or [])
        valid = [cid for cid in cited_ids if cid in known_ids]
        invalid = [cid for cid in cited_ids if cid not in known_ids]

        issues: List[str] = []
        if invalid:
            issues.append(f"Fabricated/unverifiable citation marker(s): {invalid}")

        score = len(valid) / len(cited_ids) if cited_ids else 0.5
        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=0.9 if known_ids else 0.2,  # only confident if we actually had a registry to check against
            issues=issues,
            metadata={"n_cited": len(cited_ids), "n_valid": len(valid)},
        )
        return result.as_dict()
