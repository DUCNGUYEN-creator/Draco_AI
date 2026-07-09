# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
VerifierCache
================
Caches a single verifier's result keyed by (verifier_name, claim,
evidence_hash) — finer-grained than EvidenceCache, lets an unchanged
verifier+claim+evidence triple skip re-computation even if OTHER
verifiers in the ensemble changed (e.g. a new verifier was added to
the strategy mid-session).
"""

from __future__ import annotations

from typing import Optional

from ....utils.hashing import stable_hash
from .lru import LRUCache


class VerifierCache:
    def __init__(self, max_size: int = 1024, ttl_seconds: float = 600.0) -> None:
        self._cache = LRUCache(max_size, ttl_seconds)

    def _key(self, verifier_name: str, claim: str, evidence_signature: str) -> str:
        return stable_hash(verifier_name, claim, evidence_signature)

    def get(self, verifier_name: str, claim: str, evidence_signature: str) -> Optional[dict]:
        return self._cache.get(self._key(verifier_name, claim, evidence_signature))

    def set(self, verifier_name: str, claim: str, evidence_signature: str, result: dict) -> None:
        self._cache.set(self._key(verifier_name, claim, evidence_signature), result)
