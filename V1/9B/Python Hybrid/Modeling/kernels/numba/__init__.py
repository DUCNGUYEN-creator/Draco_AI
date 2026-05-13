# DracoAI V1 — modeling/kernels/numba/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
from __future__ import annotations

__all__ = ["_register_all"]


def _register_all():
    from .. import register_kernel
    try:
        from .fused_mlp import numba_fused_swiglu
        register_kernel("fused_swiglu", numba_fused_swiglu)
    except Exception:
        pass
    try:
        from .quant_matmul import numba_int8_matmul, numba_int4_matmul
        register_kernel("int8_matmul", numba_int8_matmul)
        register_kernel("int4_matmul", numba_int4_matmul)
    except Exception:
        pass