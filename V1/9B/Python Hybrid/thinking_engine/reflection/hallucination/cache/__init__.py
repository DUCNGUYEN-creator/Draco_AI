# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.cache
==================================================
Caching layer for expensive, repeatable Hallucination subsystem work:
evidence lookups, verifier runs, calibration model state, and the
running statistics used in metrics/*. Keeps the per-request hot path
fast when the same claim/evidence pair recurs (e.g. retried turns,
repeated sub-claims across a long answer).
"""

from .lru import LRUCache
from .evidence_cache import EvidenceCache
from .verifier_cache import VerifierCache
from .calibration_cache import CalibrationCache
from .statistics_cache import StatisticsCache

__all__ = ["LRUCache", "EvidenceCache", "VerifierCache", "CalibrationCache", "StatisticsCache"]
