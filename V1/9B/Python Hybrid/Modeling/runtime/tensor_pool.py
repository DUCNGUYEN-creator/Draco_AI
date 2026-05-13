# DracoAI V1 — modeling/runtime/tensor_pool.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Thread-safe reusable NumPy workspace buffer pool."""
from __future__ import annotations
import threading
from typing import Dict
import numpy as np

__all__ = ["TensorPool"]


class TensorPool:
    def __init__(self):
        self._store: Dict[tuple, list] = {}
        self._lock   = threading.Lock()
        self._hits   = 0
        self._misses = 0

    def get(self, shape: tuple, dtype: np.dtype) -> np.ndarray:
        key = (shape, np.dtype(dtype).str)
        with self._lock:
            bucket = self._store.get(key)
            if bucket:
                self._hits += 1
                return bucket.pop()
        self._misses += 1
        return np.empty(shape, dtype=dtype)

    def put(self, arr: np.ndarray):
        key = (arr.shape, arr.dtype.str)
        with self._lock:
            self._store.setdefault(key, []).append(arr)

    def clear(self):
        with self._lock:
            self._store.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def stats(self) -> dict:
        with self._lock:
            n_bufs = sum(len(v) for v in self._store.values())
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": round(self.hit_rate, 3), "pooled_buffers": n_bufs}

    def __repr__(self) -> str:
        return f"TensorPool(hit_rate={self.hit_rate:.1%}, {self.stats()['pooled_buffers']} bufs)"