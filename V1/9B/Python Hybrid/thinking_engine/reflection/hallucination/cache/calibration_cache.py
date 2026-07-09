# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationCache
===================
Caches the EXPORTED params of a fitted calibration model — lets a
calibrator's fitted state be persisted/restored across process
restarts (e.g. serialized to disk between server runs) without
re-accumulating its whole (raw_score, label) history from scratch.
"""

from __future__ import annotations

from typing import Optional

from ..models.calibration import CalibrationModel
from .lru import LRUCache


class CalibrationCache:
    def __init__(self, max_size: int = 64) -> None:
        # No TTL — calibration state should persist indefinitely until
        # explicitly invalidated, not silently expire mid-session.
        self._cache = LRUCache(max_size, ttl_seconds=float("inf"))

    def get(self, method: str, scope_key: str = "default") -> Optional[CalibrationModel]:
        return self._cache.get(f"{method}:{scope_key}")

    def set(self, method: str, model: CalibrationModel, scope_key: str = "default") -> None:
        self._cache.set(f"{method}:{scope_key}", model)
