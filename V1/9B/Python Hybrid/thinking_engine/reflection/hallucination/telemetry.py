# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
telemetry.py
==============
Lightweight, thread-safe telemetry collector for the Hallucination
subsystem — tracks latency, cache-hit/miss rates, risk-level
distribution, and verifier-invocation counts without blocking the hot
path (all writes are lock-protected micro-increments, never I/O).

telemetry.py is the one place both pipeline/* and benchmarks/* write
to; consumers (monitoring dashboards, the StatisticsCache snapshot
refresh, DriftDetector feeds) read the snapshot without needing access
to pipeline internals.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Dict


class HallucinationTelemetry:
    """Thread-safe accumulator — safe to share across coroutines/threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_latency_ms = 0.0
        self._risk_level_counts: Dict[str, int] = defaultdict(int)
        self._verifier_call_counts: Dict[str, int] = defaultdict(int)
        self._verifier_latency_ms: Dict[str, float] = defaultdict(float)
        self._cache_hits = 0
        self._cache_misses = 0
        self._calibration_records = 0
        self._fusion_calls: Dict[str, int] = defaultdict(int)
        self._started_at = time.time()

    def record_request(self, latency_ms: float, risk_level: str) -> None:
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms
            self._risk_level_counts[risk_level] += 1

    def record_verifier(self, name: str, latency_ms: float) -> None:
        with self._lock:
            self._verifier_call_counts[name] += 1
            self._verifier_latency_ms[name] += latency_ms

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    def record_calibration(self) -> None:
        with self._lock:
            self._calibration_records += 1

    def record_fusion(self, method: str) -> None:
        with self._lock:
            self._fusion_calls[method] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._started_at
            avg_latency = (
                self._total_latency_ms / self._request_count
                if self._request_count
                else 0.0
            )
            cache_total = self._cache_hits + self._cache_misses
            cache_hit_rate = self._cache_hits / cache_total if cache_total else 0.0
            return {
                "uptime_seconds": round(uptime, 1),
                "request_count": self._request_count,
                "avg_latency_ms": round(avg_latency, 2),
                "risk_level_distribution": dict(self._risk_level_counts),
                "verifier_call_counts": dict(self._verifier_call_counts),
                "verifier_avg_latency_ms": {
                    k: round(v / self._verifier_call_counts[k], 2)
                    for k, v in self._verifier_latency_ms.items()
                    if self._verifier_call_counts[k] > 0
                },
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "calibration_records": self._calibration_records,
                "fusion_calls": dict(self._fusion_calls),
            }


# Module-level singleton — one telemetry instance per process, shared
# by every AssessmentPipeline/Assessor instantiated in that process.
_TELEMETRY = HallucinationTelemetry()


def get_telemetry() -> HallucinationTelemetry:
    return _TELEMETRY
