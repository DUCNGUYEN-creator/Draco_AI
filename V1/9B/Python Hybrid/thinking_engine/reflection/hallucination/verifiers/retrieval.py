# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
RetrievalVerifier
====================
Checks whether a claim is actually supported by the evidence that was
retrieved for it (KG triples, RAG passages, memory). This is the
single most important verifier in the ensemble: most real-world
hallucinations are claims with NO grounding evidence at all, not
claims that contradict good evidence (that's ContradictionVerifier's
job).

Scoring model
--------------
score = weighted lexical-overlap between claim and the best-matching
evidence item, scaled by that evidence's source trust_score (so a
high-trust KG triple counts for more than a low-trust scraped memory
snippet with the same lexical overlap).

No-evidence handling
----------------------
When the evidence bundle is completely empty, this verifier ABSTAINS
(low confidence) rather than confidently asserting the claim is
unsupported. This is a deliberate design choice, not an oversight: an
empty bundle usually means the retrieval INFRASTRUCTURE didn't surface
anything for this claim (e.g. a self-contained arithmetic or logic
claim that needs no external evidence at all, verifiable instead by
NumericalVerifier/SymbolicVerifier), not that the claim itself is
false. Treating "no evidence" as a confident failure signal would let
this single verifier dominate fusion (especially under noisy_or, where
one strong signal is meant to dominate) and produce false-positive
CRITICAL risk on claims that are actually correct and fully verified
by other verifiers. Compare with NumericalVerifier/SymbolicVerifier/
CitationVerifier/PlannerVerifier/ToolVerifier, which all follow the
same "abstain when there's nothing to check" convention.
"""

from __future__ import annotations

from typing import Any, Dict

from ..models.enums import EvidenceType
from ..models.enums import VerifierKind
from ..models.evidence import EvidenceBundle
from ..models.verification import VerificationResult

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "và", "là", "của", "có", "được", "này", "đó", "cho", "với", "một",
}


def _content_words(text: str) -> set:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1}


class RetrievalVerifier:
    name = "retrieval"
    kind = VerifierKind.RETRIEVAL

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        bundle: EvidenceBundle = (
            evidence if isinstance(evidence, EvidenceBundle) else EvidenceBundle(claim=claim, items=[])
        )

        if bundle.is_empty():
            # Abstain — see "No-evidence handling" in the module docstring
            # for why this must NOT be a confident failure signal.
            result = VerificationResult(
                verifier=self.name,
                kind=self.kind,
                claim=claim,
                score=0.5,
                confidence=0.05,
                issues=[],
            )
            return result.as_dict()

        claim_words = _content_words(claim)
        best_overlap = 0.0
        best_item = None
        for item in bundle.items:
            ev_words = _content_words(item.text)
            if not claim_words or not ev_words:
                continue
            overlap = len(claim_words & ev_words) / len(claim_words)
            weighted = overlap * (0.5 + 0.5 * item.trust_score)  # trust modulates, never zeroes out
            if weighted > best_overlap:
                best_overlap = weighted
                best_item = item

        score = min(best_overlap, 1.0)
        issues = []
        if score < 0.3:
            issues.append("Claim has weak lexical/semantic overlap with retrieved evidence.")
        confidence = 0.5 + 0.4 * bundle.best_trust()  # more confident when source trust is high

        result = VerificationResult(
            verifier=self.name,
            kind=self.kind,
            claim=claim,
            score=score,
            confidence=min(confidence, 0.95),
            issues=issues,
            metadata={
                "best_evidence_source": best_item.source_type.value if best_item else EvidenceType.NONE.value,
                "n_evidence_items": len(bundle.items),
            },
        )
        return result.as_dict()
