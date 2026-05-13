# DracoAI V1 — modeling/layers/norm.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""RMSNorm wrapper — delegates to ops/tensor_ops."""
from __future__ import annotations
import numpy as np
from ..ops.tensor_ops import rms_norm as _rms_norm
from ..constants import NORM_EPS

__all__ = ["RMSNorm", "rms_norm"]

# Functional alias for layer-level imports
rms_norm = _rms_norm


class RMSNorm:
    """
    Stateful RMSNorm layer — holds weight vector.
    Use for modules that prefer OOP interface over functional.
    """

    def __init__(self, d_model: int, eps: float = NORM_EPS, dtype: np.dtype = np.float32):
        self.weight = np.ones(d_model, dtype=dtype)
        self.eps    = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _rms_norm(x, self.weight, self.eps)

    def __repr__(self) -> str:
        return f"RMSNorm(d={len(self.weight)}, eps={self.eps})"