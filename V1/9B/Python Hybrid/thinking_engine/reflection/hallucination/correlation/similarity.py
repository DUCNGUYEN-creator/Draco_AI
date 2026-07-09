# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SimilarityScorer
===================
Jaccard word-overlap similarity between two evidence texts — the
dependency-free building block every correlation/* module uses to
decide "are these two pieces of evidence basically saying the same
thing". Deliberately simple (no embeddings) so the Hallucination
subsystem has zero hard external-model dependencies; swap in a real
embedding-cosine scorer here later without touching callers.
"""

from __future__ import annotations


class SimilarityScorer:
    def jaccard(self, a: str, b: str) -> float:
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def is_near_duplicate(self, a: str, b: str, threshold: float = 0.82) -> bool:
        return self.jaccard(a, b) >= threshold
