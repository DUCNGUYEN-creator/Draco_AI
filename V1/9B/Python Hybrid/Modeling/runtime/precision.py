# DracoAI V1 — modeling/runtime/precision.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DynamicPrecisionManager — advisory dtype switching.
Caller must act on current_dtype; this manager never mutates model weights.
"""
from __future__ import annotations
import numpy as np

__all__ = ["DynamicPrecisionManager"]


class DynamicPrecisionManager:
    def __init__(self, overflow_thresh: float = 40.0, up_thresh: float = 0.05,
                 down_thresh: float = 0.005, alpha: float = 0.1,
                 initial_dtype: np.dtype = np.float16):
        self._overflow_thresh = overflow_thresh
        self._up_thresh       = up_thresh
        self._down_thresh     = down_thresh
        self._alpha           = alpha
        self._ema             = 0.0
        self._current_dtype   = np.dtype(initial_dtype)
        self._n_upgrades      = self._n_downgrades = self._n_steps = 0

    @property
    def current_dtype(self) -> np.dtype:
        return self._current_dtype

    def update(self, logits: np.ndarray) -> np.dtype:
        self._n_steps += 1
        if logits.size == 0:
            return self._current_dtype
        of = float((np.abs(logits) > self._overflow_thresh).mean())
        self._ema = self._alpha * of + (1 - self._alpha) * self._ema
        if self._current_dtype == np.float16 and self._ema > self._up_thresh:
            self._current_dtype = np.dtype(np.float32); self._n_upgrades += 1
        elif self._current_dtype == np.float32 and self._ema < self._down_thresh:
            self._current_dtype = np.dtype(np.float16); self._n_downgrades += 1
        return self._current_dtype

    def status(self) -> dict:
        return dict(current_dtype=str(self._current_dtype),
                    overflow_ema=round(self._ema, 5),
                    n_upgrades=self._n_upgrades,
                    n_downgrades=self._n_downgrades, steps=self._n_steps)

    def reset(self):
        self._ema = 0.0; self._n_upgrades = self._n_downgrades = self._n_steps = 0

    def __repr__(self) -> str:
        s = self.status()
        return (f"DynamicPrecisionManager(dtype={s['current_dtype']}, "
                f"ema={s['overflow_ema']:.4f}, upgrades={s['n_upgrades']})")