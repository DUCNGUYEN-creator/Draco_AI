# DracoAI V1 — modeling/quant/quant_linear.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Model-level weight quantization helper.
quantize_model_weights() replaces attention/FFN weights in-place.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from .int4 import QuantizedLinear

if TYPE_CHECKING:
    from ..transformer import DracoTransformerV1

__all__ = ["quantize_model_weights"]


def quantize_model_weights(
    model:      "DracoTransformerV1",
    quant:      str,
    group_size: int = 128,
) -> None:
    """Replace all float weight matrices in-place with QuantizedLinear."""
    if quant not in ("int8", "int4"):
        raise ValueError(f"quant must be 'int8' or 'int4', got {quant!r}")

    for blk in model.blocks:
        # Attention weights
        for attr in ("W_q", "W_k", "W_v", "W_o"):
            W = getattr(blk.attn, attr)
            if isinstance(W, QuantizedLinear):
                continue
            ql = QuantizedLinear.from_float(W.T, quant=quant, group_size=group_size)
            ql._transposed = True
            setattr(blk.attn, attr, ql)

        # FFN expert weights
        for exp in list(blk.moe.experts) + [blk.moe.shared]:
            for attr in ("W_g", "W_u", "W_d"):
                W = getattr(exp, attr)
                if isinstance(W, QuantizedLinear):
                    continue
                ql = QuantizedLinear.from_float(W.T, quant=quant, group_size=group_size)
                ql._transposed = True
                setattr(exp, attr, ql)

        # Invalidate stacked weight cache in MoE
        blk.moe._invalidate_stacked()