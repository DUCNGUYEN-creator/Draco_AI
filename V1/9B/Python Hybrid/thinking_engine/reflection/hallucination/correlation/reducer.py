# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EvidenceReducer
==================
Top-level orchestrator combining EvidenceDeduplicator (evidence-level)
with a verifier-signal discount derived from VerifierDependencyGraph
(verifier-level) — the single call assessor.py / pipeline/* makes to
go from "raw evidence + raw verifier results" to "decorrelated inputs
ready for fusion".
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..models.evidence import EvidenceBundle
from .deduplication import EvidenceDeduplicator
from .dependency import VerifierDependencyGraph


class EvidenceReducer:
    def __init__(self, similarity_threshold: float = 0.82) -> None:
        self._dedup = EvidenceDeduplicator(threshold=similarity_threshold)
        self._dep_graph = VerifierDependencyGraph()

    def reduce_evidence(self, bundle: EvidenceBundle) -> Tuple[EvidenceBundle, int]:
        original_n = len(bundle.items)
        deduped = self._dedup.deduplicate(bundle)
        return deduped, original_n - len(deduped.items)

    def verifier_discounts(self, verification_results: List[dict]) -> Dict[str, float]:
        """Returns a per-verifier multiplier in (0, 1] — verifiers with
        more total correlation mass to other PRESENT verifiers get
        discounted more, since their signal is more likely to be
        redundant with something else already in the ensemble."""
        present = [r.get("verifier", "") for r in verification_results if r.get("confidence", 0.0) > 0.1]
        adj = self._dep_graph.build(present)
        discounts: Dict[str, float] = {}
        for v in present:
            mass = self._dep_graph.total_correlation_mass(adj, v)
            discounts[v] = max(0.5, 1.0 - 0.25 * mass)
        return discounts
