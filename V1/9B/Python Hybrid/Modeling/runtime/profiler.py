# DracoAI V1 — modeling/runtime/profiler.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Lightweight per-step inference profiler."""
from __future__ import annotations
import time
from typing import List
import numpy as np

__all__ = ["InferenceProfiler"]


class InferenceProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self._fwd_times:     List[float] = []
        self._spec_accept    = self._spec_reject = self._n_tokens = 0
        self._start_wall     = self._end_wall = 0.0
        self._escalate_count = 0
        self._peak_mem_mb    = 0.0

    def start_session(self):
        self.reset(); self._start_wall = time.perf_counter()

    def record_forward(self, elapsed_s: float):
        self._fwd_times.append(elapsed_s * 1000.0)
        try:
            import resource as _res, sys as _sys
            rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
            mb  = rss / 1024 if _sys.platform != "darwin" else rss / (1024 * 1024)
            if mb > self._peak_mem_mb:
                self._peak_mem_mb = mb
        except Exception:
            pass

    def record_spec_accept(self): self._spec_accept    += 1
    def record_spec_reject(self): self._spec_reject    += 1
    def record_tokens(self, n):   self._n_tokens       += n
    def record_escalate(self):    self._escalate_count += 1
    def end_session(self):        self._end_wall = time.perf_counter()

    def summary(self) -> dict:
        wall = max(self._end_wall - self._start_wall, 1e-9)
        fwd  = self._fwd_times
        return dict(
            total_tokens=self._n_tokens,
            wall_time_s=round(wall, 3),
            tokens_per_sec=round(self._n_tokens / wall, 2),
            n_forward_calls=len(fwd),
            avg_fwd_ms=round(float(np.mean(fwd)), 2) if fwd else 0.0,
            p95_fwd_ms=round(float(np.percentile(fwd, 95)), 2) if fwd else 0.0,
            spec_accept=self._spec_accept,
            spec_reject=self._spec_reject,
            spec_accept_rate=round(
                self._spec_accept / max(1, self._spec_accept + self._spec_reject), 3),
            peak_mem_mb=round(self._peak_mem_mb, 1),
            snap_escalations=self._escalate_count,
            reject_rate=round(self._spec_reject / max(1, self._n_tokens), 3),
        )

    def __repr__(self) -> str:
        s = self.summary()
        return (f"InferenceProfiler | {s['total_tokens']} tok | "
                f"{s['tokens_per_sec']} tok/s | avg_fwd={s['avg_fwd_ms']}ms | "
                f"spec={s['spec_accept_rate']:.1%} | mem={s['peak_mem_mb']}MB")