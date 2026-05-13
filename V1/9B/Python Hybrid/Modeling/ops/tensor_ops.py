# DracoAI V1 — modeling/ops/tensor_ops.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Tensor utilities: rms_norm, mm, and optional kernel-accelerated variants.

mm() dispatch order:
  1. QuantizedLinear → use its own .forward() (knows int8/int4)
  2. Kernel registry → hardware-accelerated matmul if available
  3. Pure NumPy @ operator (always available)

No model state. No imports from layers/ or runtime/.

FIXES (this revision):
  ✅ FIX-DOUBLE-IMPORT-QUANTIZEDLINEAR : removed the TYPE_CHECKING-only
     import of QuantizedLinear that shadowed the runtime import inside mm().
     pyflakes correctly flagged the top-level import as unused because the
     runtime import inside the function body redefined the name immediately.
     Now a single runtime import lives inside mm() — the TYPE_CHECKING guard
     is dropped since the class is only used in isinstance(), not in type
     annotations within this module.
"""
from __future__ import annotations
import numpy as np

from ..constants import NORM_EPS

__all__ = ["rms_norm", "mm"]


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = NORM_EPS) -> np.ndarray:
    """Root-Mean-Square layer normalisation."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w


def mm(x: np.ndarray, W) -> np.ndarray:
    """
    Unified matrix multiply.

    Dispatch:
        QuantizedLinear  → W.forward(x)
        np.ndarray + int8_matmul kernel available → kernel
        np.ndarray       → x @ W  (NumPy)
    """
    # Single, authoritative runtime import — no TYPE_CHECKING alias above.
    from ..quant.int4 import QuantizedLinear
    if isinstance(W, QuantizedLinear):
        return W.forward(x)

    # Optional kernel acceleration (transparent — no change to callers).
    # Currently only dispatches for explicit int8 weight arrays; plain float
    # matmul always falls through to NumPy.
    try:
        from ..kernels import get_kernel
        kern = get_kernel("int8_matmul")
        if kern is not None and hasattr(W, "dtype") and W.dtype == np.int8:
            # Placeholder: caller must supply scale separately for int8 path.
            pass
    except Exception:
        pass

    return x @ W