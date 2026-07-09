# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Lightweight call-count + cumulative-time profiler — no external deps.
Used by telemetry.py to track per-verifier / per-stage cost without
pulling in cProfile overhead on every request."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Dict


class InferenceProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_ms: Dict[str, float] = defaultdict(float)

    def record(self, label: str, duration_ms: float) -> None:
        with self._lock:
            self._counts[label] += 1
            self._total_ms[label] += duration_ms

    def report(self) -> Dict[str, dict]:
        with self._lock:
            return {
                label: {
                    "count": self._counts[label],
                    "total_ms": round(self._total_ms[label], 3),
                    "avg_ms": round(self._total_ms[label] / self._counts[label], 3)
                    if self._counts[label]
                    else 0.0,
                }
                for label in self._counts
            }
