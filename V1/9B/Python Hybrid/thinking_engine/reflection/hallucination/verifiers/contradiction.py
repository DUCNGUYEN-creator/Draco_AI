# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ContradictionVerifier
========================
Distinct from RetrievalVerifier (no support found) — this verifier
flags claims that ACTIVELY CONTRADICT retrieved evidence, which is a
much stronger hallucination signal than mere lack-of-support. Looks
for negation-flip patterns and antonym pairs between the claim and the
best-matching evidence item.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..models.enums import VerifierKind
from ..models.evidence import EvidenceBundle
from ..models.verification import VerificationResult

_NEGATION_MARKERS = ["không", "chẳng", "not", "no", "never", "isn't", "wasn't", "doesn't"]

_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("tăng", "giảm"), ("increase", "decrease"), ("true", "false"),
    ("đúng", "sai"), ("more", "less"), ("nhiều", "ít"),
    ("trước", "sau"), ("before", "after"), ("lớn", "nhỏ"), ("bigger", "smaller"),
]


def _has_negation(text: str) -> bool:
    tl = text.lower()
    return any(re.search(rf"\b{re.escape(m)}\b", tl) for m in _NEGATION_MARKERS)


class ContradictionVerifier:
    name = "contradiction"
    kind = VerifierKind.CONTRADICTION

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        bundle: EvidenceBundle = (
            evidence if isinstance(evidence, EvidenceBundle) else EvidenceBundle(claim=claim, items=[])
        )
        if bundle.is_empty():
            # Nothing to contradict — neutral, not a contradiction signal.
            result = VerificationResult(
                verifier=self.name, kind=self.kind, claim=claim,
                score=0.7, confidence=0.3, issues=[],
            )
            return result.as_dict()

        claim_l = claim.lower()
        contradictions: List[str] = []

        for item in bundle.items:
            ev_l = item.text.lower()

            # Negation-flip: claim asserts X negated, evidence asserts X plain (or vice versa)
            if _has_negation(claim_l) != _has_negation(ev_l):
                shared = set(claim_l.split()) & set(ev_l.split())
                meaningful_shared = {w for w in shared if len(w) > 3}
                if len(meaningful_shared) >= 2:
                    contradictions.append(
                        f"Negation mismatch with evidence (shared terms: {sorted(meaningful_shared)[:3]})"
                    )

            # Antonym pairs both present across claim vs evidence
            for a, b in _ANTONYM_PAIRS:
                if (a in claim_l and b in ev_l) or (b in claim_l and a in ev_l):
                    contradictions.append(f"Antonym conflict: '{a}' vs '{b}' across claim/evidence")

        n_contradictions = len(contradictions)
        # More contradictions => lower support score, higher detection confidence
        score = max(0.0, 1.0 - 0.35 * n_contradictions)
        confidence = min(0.5 + 0.15 * n_contradictions, 0.95)

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=confidence,
            issues=contradictions[:5],
            metadata={"n_contradictions": n_contradictions},
        )
        return result.as_dict()
