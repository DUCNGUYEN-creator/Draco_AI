# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EvidenceDeduplicator
=======================
Collapses an EvidenceBundle's near-duplicate items down to one
representative per correlation group (picking the highest-trust member
of each group) — this is what RetrievalVerifier should run evidence
through BEFORE scoring, so 5 copies of the same RAG chunk retrieved
under slightly different queries don't masquerade as "5 independent
sources confirm this claim".
"""

from __future__ import annotations

from typing import List

from ..models.evidence import Evidence, EvidenceBundle
from .connected_components import ConnectedComponentsClusterer


class EvidenceDeduplicator:
    def __init__(self, threshold: float = 0.82) -> None:
        self._clusterer = ConnectedComponentsClusterer(threshold=threshold)

    def deduplicate(self, bundle: EvidenceBundle) -> EvidenceBundle:
        if bundle.is_empty():
            return bundle
        texts = bundle.texts()
        groups = self._clusterer.correlate(texts)

        kept: List[Evidence] = []
        for group in groups:
            members = [bundle.items[i] for i in group.member_indices]
            best = max(members, key=lambda e: e.trust_score)
            kept.append(best)
        return EvidenceBundle(claim=bundle.claim, items=kept)
