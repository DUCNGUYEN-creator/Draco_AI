# DracoAI V1 — modeling/kernels/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Kernel registry — optional hardware-accelerated implementations.

ops/tensor_ops.py and layers/mlp.py query this registry via get_kernel().
If a kernel is unavailable (missing package, no GPU), get_kernel()
returns None and the caller falls back to pure NumPy.

Architecture contract:
  - Kernels accept np.ndarray, return np.ndarray.
  - Kernels have NO knowledge of layers/, runtime/, or kv_cache/.
  - Import failures are silently swallowed — NumPy is always the fallback.

Registered kernels (populated at import time):
  fused_attention   : Triton scaled-dot-product attention (GPU)
  fused_swiglu      : Triton/Numba fused SwiGLU MLP (GPU/CPU JIT)
  int8_matmul       : Triton/Numba INT8 weight-only matmul
  int4_matmul       : Triton/Numba INT4 weight-only matmul
  ternary_matmul    : Triton addition-only ternary matmul (NEW)
"""
from __future__ import annotations
import logging
from typing import Callable, Optional

__all__ = ["get_kernel", "register_kernel", "list_kernels"]

logger = logging.getLogger(__name__)

# Registry: op_name → callable
_REGISTRY: dict = {}


def register_kernel(op_name: str, fn: Callable):
    """Register a kernel for op_name. Last write wins."""
    _REGISTRY[op_name] = fn
    logger.debug("[KernelRegistry] registered %s", op_name)


def get_kernel(op_name: str) -> Optional[Callable]:
    """Return the registered kernel or None if unavailable."""
    return _REGISTRY.get(op_name)


def list_kernels() -> list:
    return list(_REGISTRY.keys())


# ── Auto-load available kernels on import ─────────────────────────────
def _load_backends():
    try:
        from .triton import _register_all as _triton_reg
        _triton_reg()
    except Exception as e:
        logger.debug("[KernelRegistry] Triton backend not loaded: %s", e)
    try:
        from .numba import _register_all as _numba_reg
        _numba_reg()
    except Exception as e:
        logger.debug("[KernelRegistry] Numba backend not loaded: %s", e)


_load_backends()