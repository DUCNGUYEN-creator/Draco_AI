# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
EvidenceCache
================
Caches EvidenceBundles keyed by claim text — avoids re-running
RAG/KG/memory retrieval for the same claim text appearing more than
once in a single response (common with repeated sub-claims) or across
a retry/refine loop.
"""

from __future__ import annotations

from typing import Optional

from ..models.evidence import EvidenceBundle
from .lru import LRUCache


class EvidenceCache:
    def __init__(self, max_size: int = 512, ttl_seconds: float = 600.0) -> None:
        self._cache = LRUCache(max_size, ttl_seconds)

    def get(self, claim: str) -> Optional[EvidenceBundle]:
        return self._cache.get(claim.strip().lower())

    def set(self, claim: str, bundle: EvidenceBundle) -> None:
        self._cache.set(claim.strip().lower(), bundle)
