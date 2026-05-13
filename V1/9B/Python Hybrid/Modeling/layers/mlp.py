# DracoAI V1 — modeling/layers/mlp.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExpertFFN — SwiGLU feed-forward expert.

Kernel dispatch order (transparent to callers):
  1. Triton fused_swiglu (GPU)
  2. Numba fused_swiglu (CPU JIT)
  3. NumPy (always)
"""
from __future__ import annotations
import math
import numpy as np
from ..ops.tensor_ops import mm
from ..ops.activation  import silu

__all__ = ["ExpertFFN"]


class ExpertFFN:
    """
    SwiGLU expert: output = (silu(x @ W_g) * (x @ W_u)) @ W_d
    """

    def __init__(self, d_model: int, d_ff: int):
        scale    = 1.0 / math.sqrt(d_model)
        self.W_g = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_u = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_d = np.random.randn(d_ff, d_model).astype(np.float32) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Try fused kernel (Triton > Numba > NumPy)
        try:
            from ..kernels import get_kernel
            from ..quant.int4 import QuantizedLinear
            kern = get_kernel("fused_swiglu")
            if kern is not None and not any(
                    isinstance(w, QuantizedLinear)
                    for w in (self.W_g, self.W_u, self.W_d)):
                return kern(x.astype(np.float32), self.W_g, self.W_u, self.W_d)
        except Exception:
            pass
        # NumPy path
        gate = mm(x, self.W_g)
        gate = silu(gate)
        return mm(gate * mm(x, self.W_u), self.W_d)

    def _break_symmetry(self, scale: float = 1e-3):
        self.W_g += np.random.randn(*self.W_g.shape).astype(np.float32) * scale
        self.W_u += np.random.randn(*self.W_u.shape).astype(np.float32) * scale