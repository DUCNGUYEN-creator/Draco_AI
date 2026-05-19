# DracoAI V1 — modeling/kernels/triton/__init__.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Triton GPU kernel package. Only loaded when triton + CUDA are available."""
from __future__ import annotations

__all__ = ["_register_all"]


def _register_all():
    """Register all Triton kernels into the central registry."""
    from .. import register_kernel
    try:
        from .fused_attention import triton_fused_attention
        register_kernel("fused_attention", triton_fused_attention)
    except Exception:
        pass
    try:
        from .fused_mlp import triton_fused_swiglu
        register_kernel("fused_swiglu", triton_fused_swiglu)
    except Exception:
        pass
    try:
        from .quant_matmul import triton_int8_matmul, triton_int4_matmul
        register_kernel("int8_matmul", triton_int8_matmul)
        register_kernel("int4_matmul", triton_int4_matmul)
    except Exception:
        pass
    try:
        from .ternary_matmul import triton_ternary_matmul
        register_kernel("ternary_matmul", triton_ternary_matmul)
    except Exception:
        pass