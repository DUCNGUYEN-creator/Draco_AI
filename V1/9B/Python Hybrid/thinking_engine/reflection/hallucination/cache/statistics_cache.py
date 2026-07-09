# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
StatisticsCache
==================
Caches metrics.verifier_score.VerifierScoreTracker / RunningStats
snapshots so benchmarks/* and telemetry.py don't need direct access to
the live, lock-protected tracker objects — they read a recent
point-in-time snapshot instead, avoiding lock contention with the
per-request hot path.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class StatisticsCache:
    def __init__(self, refresh_interval_seconds: float = 30.0) -> None:
        self.refresh_interval_seconds = refresh_interval_seconds
        self._snapshot: Dict[str, Any] = {}
        self._last_refresh: float = 0.0

    def get_or_refresh(self, compute_fn) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_refresh >= self.refresh_interval_seconds or not self._snapshot:
            self._snapshot = compute_fn()
            self._last_refresh = now
        return self._snapshot
