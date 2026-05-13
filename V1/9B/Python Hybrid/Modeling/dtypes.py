# DracoAI V1 — modeling/dtypes.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Dtype policy — queries device.py to select the best compute dtype.

Rules:
  - GPU (CUDA) available : prefer float16 (bfloat16 not available in NumPy).
  - CPU only             : float16 on modern hardware (AVX2), else float32.
  - Caller can override  : set_compute_dtype(np.float32) at startup.

This module is the SINGLE source of truth for COMPUTE_DTYPE.
config.py re-exports it but MUST NOT redefine it — it imports and
re-exports get_compute_dtype() so all callers see updates from
set_compute_dtype().

FIXES (this revision):
  ✅ FIX-COMPUTE-DTYPE-SINGLE-SOURCE : COMPUTE_DTYPE is no longer a frozen
     module-level alias. It is replaced by get_compute_dtype() which always
     returns the live _COMPUTE_DTYPE.  config.py no longer calls
     _detect_compute_dtype() itself — it delegates entirely to this module.
     This ensures set_compute_dtype() is visible everywhere.
"""
from __future__ import annotations
import logging
import numpy as np

__all__ = [
    "get_compute_dtype", "set_compute_dtype",
    "safe_cast", "is_low_precision",
]

logger = logging.getLogger(__name__)

# Module-level mutable state (intentional — single override point)
_COMPUTE_DTYPE: np.dtype = np.dtype(np.float32)
_DTYPE_LOCKED:  bool     = False


def _select_dtype() -> np.dtype:
    """Auto-select best dtype based on hardware capabilities."""
    try:
        from .device import get_capability
        cap = get_capability()
        if cap.has_cuda or cap.has_avx2:
            return np.dtype(np.float16)
    except Exception:
        pass
    return np.dtype(np.float32)


def _init():
    global _COMPUTE_DTYPE
    _COMPUTE_DTYPE = _select_dtype()
    logger.debug("[DracoAI] COMPUTE_DTYPE auto-selected: %s", _COMPUTE_DTYPE)


_init()


def get_compute_dtype() -> np.dtype:
    """Return the current compute dtype (always fresh, never stale)."""
    return _COMPUTE_DTYPE


def set_compute_dtype(dtype: np.dtype, lock: bool = False):
    """
    Override the compute dtype globally.

    lock=True: prevents further overrides (useful in test suites).
    All importers that use get_compute_dtype() will see the new value
    immediately.  Module-level ``COMPUTE_DTYPE`` aliases in other files
    must use get_compute_dtype() — not a frozen copy.
    """
    global _COMPUTE_DTYPE, _DTYPE_LOCKED
    if _DTYPE_LOCKED:
        logger.warning(
            "[DracoAI] COMPUTE_DTYPE is locked at %s — ignoring set_compute_dtype(%s)",
            _COMPUTE_DTYPE, dtype)
        return
    _COMPUTE_DTYPE = np.dtype(dtype)
    _DTYPE_LOCKED  = lock
    logger.info("[DracoAI] COMPUTE_DTYPE set to %s (locked=%s)", _COMPUTE_DTYPE, lock)


def safe_cast(arr: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """
    Cast arr to dtype (default: current COMPUTE_DTYPE) without copying if
    already correct.  Always returns a float array (never int or bool).
    Uses the live _COMPUTE_DTYPE so it reflects any set_compute_dtype() call.
    """
    target = np.dtype(dtype) if dtype is not None else _COMPUTE_DTYPE
    if arr.dtype == target:
        return arr
    return arr.astype(target)


def is_low_precision(dtype: np.dtype | None = None) -> bool:
    """True when compute dtype is float16 (lower memory, potential overflow)."""
    d = np.dtype(dtype) if dtype is not None else _COMPUTE_DTYPE
    return d == np.float16