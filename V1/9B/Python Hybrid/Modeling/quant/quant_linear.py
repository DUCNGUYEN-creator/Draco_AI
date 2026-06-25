# DracoAI V1 — modeling/quant/quant_linear.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Model-level weight quantization helper.

quantize_model_weights() replaces attention/FFN weights in-place with the
requested quantization format.

NEW in this revision:
  ✅ FEAT-TERNARY-QUANT : quant='ternary' mode converts MoE expert FFN
     weights (W_g, W_u, W_d) to TernaryLinear.  Attention weights (Q, K, V, O)
     and the MoE Router are kept at full precision (RED zone policy).
     Shared expert is also ternarized (it is an FFN, same GREEN zone).
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
    """
    Replace weight matrices in-place with the specified quantization format.

    Supported quant values
    ──────────────────────
    'int8'    : Symmetric INT8 weight-only quant (attention + FFN).
    'int4'    : Asymmetric INT4 per-group quant  (attention + FFN).
    'ternary' : BitNet 1.58b ternary {-1,0,+1} for FFN (GREEN zone only).
                Attention weights stay float32 (RED zone policy).

    The 'ternary' mode follows the Hybrid Ternary-Dense strategy:
      • MoE Expert W_g, W_u, W_d → TernaryLinear
      • Shared Expert W_g, W_u, W_d → TernaryLinear
      • Attention W_q, W_k, W_v, W_o → unchanged (FP32/FP16)
      • MoE Router W_router → unchanged (FP32)
      • Norm weights → unchanged (FP32)
    """
    if quant not in ("int8", "int4", "ternary"):
        raise ValueError(f"quant must be 'int8', 'int4', or 'ternary', got {quant!r}")

    if quant == "ternary":
        _ternarize_model_weights(model)
        return

    # INT8 / INT4 path (attention + FFN)
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


def _ternarize_model_weights(model: "DracoTransformerV1") -> None:
    """
    Apply ternary quantization to FFN Expert weights only (GREEN zone).

    Attention, Router, Embedding, Norm weights are left untouched.
    """
    from .ternary_linear import TernaryLinear
    import numpy as np

    for blk in model.blocks:
        for exp in list(blk.moe.experts) + [blk.moe.shared]:
            for attr in ("W_g", "W_u", "W_d"):
                W = getattr(exp, attr)
                # Skip if already ternary or quantized
                if isinstance(W, (TernaryLinear, QuantizedLinear)):
                    continue
                if not isinstance(W, np.ndarray):
                    continue
                # Transpose: ExpertFFN stores (in, out); TernaryLinear expects (out, in)
                tl = TernaryLinear.from_float(W.T)
                setattr(exp, attr, tl)
            # Mark expert as ternary so forward dispatches to addition-only path
            exp._ternary = True

        # Invalidate stacked weight cache (ternary weights can't use einsum fast path)
        blk.moe._invalidate_stacked()