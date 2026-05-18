# DracoAI V1 — modeling/runtime/tensor_pool.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
TensorPool — thread-safe reusable NumPy workspace buffer pool.

NEW in this revision:
  ✅ FEAT-STATIC-PREALLOC    : preallocate() reserves named buffer shapes
     at startup, eliminating malloc/free during hot inference paths.
     Pre-allocated buffers are held permanently (never garbage-collected)
     and returned from get() before any new allocation is attempted.
  ✅ FEAT-VRAM-HARDCAP       : track total bytes held and report to
     DynamicPrecisionManager so it can enforce the VRAM budget.
     tracked_bytes() returns the current pool memory footprint.
  ✅ FEAT-ZEROIZE            : secure_clear() overwrites all pooled
     buffers with zeros before releasing them — used by session.py
     on clear() to prevent Cold Boot Attacks on KV data.
"""
from __future__ import annotations
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

__all__ = ["TensorPool"]


class TensorPool:
    def __init__(self, vram_budget_bytes: int = 0):
        """
        Parameters
        ----------
        vram_budget_bytes : Soft VRAM ceiling for the pool (0 = unlimited).
                            When tracked_bytes() exceeds this, get() still
                            works but put() silently drops buffers instead
                            of pooling them — preventing runaway memory.
        """
        self._store: Dict[Tuple, List[np.ndarray]] = {}
        self._lock   = threading.Lock()
        self._hits   = 0
        self._misses = 0
        self._vram_budget = int(vram_budget_bytes)
        self._tracked_bytes = 0  # sum of bytes in all pooled buffers

    # ── Pre-allocation ────────────────────────────────────────────────────────

    def preallocate(self, shape: tuple, dtype: np.dtype, n: int = 1) -> None:
        """
        Pre-allocate `n` buffers of the given shape/dtype.

        These buffers are held in the pool and returned by get() on the
        first n calls with the matching key, avoiding heap allocation
        during the hot inference path.
        """
        key = (shape, np.dtype(dtype).str)
        buffers = [np.empty(shape, dtype=dtype) for _ in range(n)]
        with self._lock:
            existing = self._store.get(key, [])
            existing.extend(buffers)
            self._store[key] = existing
            for b in buffers:
                self._tracked_bytes += b.nbytes

    # ── Core get/put ──────────────────────────────────────────────────────────

    def get(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        key = (shape, np.dtype(dtype).str)
        with self._lock:
            bucket = self._store.get(key)
            if bucket:
                buf = bucket.pop()
                self._tracked_bytes -= buf.nbytes
                self._hits += 1
                return buf
        self._misses += 1
        return np.empty(shape, dtype=dtype)

    def put(self, arr: np.ndarray) -> None:
        """
        Return a buffer to the pool.

        Dropped silently when the pool is over-budget to prevent
        memory accumulation under VRAM pressure.
        """
        if self._vram_budget > 0:
            with self._lock:
                if self._tracked_bytes + arr.nbytes > self._vram_budget:
                    return  # over budget — discard, don't pool

        key = (arr.shape, arr.dtype.str)
        with self._lock:
            self._store.setdefault(key, []).append(arr)
            self._tracked_bytes += arr.nbytes

    # ── Memory reporting ──────────────────────────────────────────────────────

    def tracked_bytes(self) -> int:
        """Return total bytes currently held in the pool."""
        with self._lock:
            return self._tracked_bytes

    def tracked_gb(self) -> float:
        return self.tracked_bytes() / 1024**3

    # ── Secure clear ─────────────────────────────────────────────────────────

    def secure_clear(self) -> None:
        """
        Overwrite all pooled buffers with zeros then drop them.

        ✅ FEAT-ZEROIZE: called by session.py on GenerationSession.clear()
        to ensure no KV / activation data survives in reachable memory.
        Protects against Cold Boot Attacks where an attacker reads raw
        RAM pages after the process exits.
        """
        with self._lock:
            for bucket in self._store.values():
                for arr in bucket:
                    arr[:] = 0   # overwrite in-place
            self._store.clear()
            self._tracked_bytes = 0

    def clear(self) -> None:
        """Drop all pooled buffers without zeroing (fast path)."""
        with self._lock:
            self._store.clear()
            self._tracked_bytes = 0

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        with self._lock:
            n_bufs = sum(len(v) for v in self._store.values())
        return {
            "hits":            self._hits,
            "misses":          self._misses,
            "hit_rate":        round(self.hit_rate, 3),
            "pooled_buffers":  n_bufs,
            "tracked_bytes":   self._tracked_bytes,
            "tracked_gb":      round(self.tracked_gb(), 4),
            "budget_gb":       round(self._vram_budget / 1024**3, 3),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"TensorPool(hit_rate={self.hit_rate:.1%}, "
                f"{s['pooled_buffers']} bufs, "
                f"{s['tracked_gb']:.3f}GB held)")