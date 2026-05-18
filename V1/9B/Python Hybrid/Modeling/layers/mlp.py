# DracoAI V1 — modeling/layers/mlp.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExpertFFN — SwiGLU feed-forward expert.

Kernel dispatch order (transparent to callers):
  1. Triton ternary_matmul (GPU, 1.58-bit Expert weights)  ← NEW
  2. Triton fused_swiglu   (GPU, FP16 Expert weights)
  3. Numba fused_swiglu    (CPU JIT, FP16)
  4. NumPy                 (always available)

Ternary mode (V1 Hybrid strategy):
  When W_g / W_u / W_d are TernaryLinear instances, the forward pass
  uses the addition-only kernel — no float multiplications in the inner
  loop.  This reduces:
    • weight bandwidth  by ~8× vs FP16
    • energy per token  by ~40–60% (no multiplier units activated)

  Ternary mode is opt-in: set ternary=True when constructing ExpertFFN
  or call ExpertFFN.ternarize() after creation.

Activation sparsity (PowerInfer-style, V1 optional):
  When a SparsityPredictor is attached, gate neurons predicted inactive
  are zeroed before the W_d projection — skipping those matrix-vector
  products.  Effective sparsity: 20–45% typical, depending on input.

Architecture note (GREEN/RED zones):
  ExpertFFN is in the GREEN zone: ternary weights are safe here because
  MoE has many experts that collectively cover the information space.
  Attention (Q/K/V/O), Router, Embedding, and Norm remain FP16/FP32.
"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np

from ..ops.tensor_ops import mm
from ..ops.activation  import silu

__all__ = ["ExpertFFN"]


class ExpertFFN:
    """
    SwiGLU expert: output = (silu(x @ W_g) * (x @ W_u)) @ W_d

    Supports three weight modes (transparent to callers):
      • FP32/FP16 ndarray  — standard float matmul
      • QuantizedLinear    — INT8/INT4 weight-only quant
      • TernaryLinear      — 1.58-bit ternary, addition-only forward
    """

    def __init__(self, d_model: int, d_ff: int, ternary: bool = False):
        scale    = 1.0 / math.sqrt(d_model)
        self.d_model  = d_model
        self.d_ff     = d_ff
        self._ternary = ternary

        # Weight matrices; may be replaced by TernaryLinear after init.
        self.W_g = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_u = np.random.randn(d_model, d_ff).astype(np.float32) * scale
        self.W_d = np.random.randn(d_ff, d_model).astype(np.float32) * scale

        # Optional activation sparsity predictor (PowerInfer-style)
        self._sparsity_pred = None   # SparsityPredictor | None

        if ternary:
            self.ternarize()

    # ── Ternary conversion ────────────────────────────────────────────────────

    def ternarize(self) -> "ExpertFFN":
        """
        Convert all three weight matrices to TernaryLinear in-place.

        Safe to call multiple times (idempotent — skips already-ternary
        weights).  Returns self for chaining.
        """
        from ..quant.ternary_linear import TernaryLinear
        for attr in ("W_g", "W_u", "W_d"):
            W = getattr(self, attr)
            if not isinstance(W, TernaryLinear):
                setattr(self, attr, TernaryLinear.from_float(
                    W if isinstance(W, np.ndarray) else W.dequantize()
                ))
        self._ternary = True
        return self

    def attach_sparsity_predictor(self, predictor) -> "ExpertFFN":
        """Attach a SparsityPredictor for gate activation skipping."""
        self._sparsity_pred = predictor
        return self

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        SwiGLU forward.  Dispatch to the best available kernel.

        x : (batch_or_seq, d_model) float32
        returns : (batch_or_seq, d_model) float32
        """
        from ..quant.int4 import QuantizedLinear
        try:
            from ..quant.ternary_linear import TernaryLinear
            _has_ternary = True
        except ImportError:
            _has_ternary = False

        # ── Ternary path (addition-only) ──────────────────────────────
        if (
            _has_ternary
            and isinstance(self.W_g, TernaryLinear)
            and isinstance(self.W_u, TernaryLinear)
            and isinstance(self.W_d, TernaryLinear)
        ):
            return self._ternary_forward(x.astype(np.float32))

        # ── Fused kernel path (Triton / Numba) ────────────────────────
        try:
            from ..kernels import get_kernel
            kern = get_kernel("fused_swiglu")
            if kern is not None and not any(
                    isinstance(w, (QuantizedLinear,))
                    for w in (self.W_g, self.W_u, self.W_d)):
                return kern(x.astype(np.float32), self.W_g, self.W_u, self.W_d)
        except Exception:
            pass

        # ── NumPy path ────────────────────────────────────────────────
        return self._numpy_forward(x)

    def _ternary_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Ternary SwiGLU: addition-only gate and up projections.
        Optional activation sparsity skip on gate.
        """
        gate = self.W_g.forward(x)          # (seq, d_ff)
        gate = silu(gate)

        # Optional: zero predicted-inactive neurons (sparsity skip)
        if self._sparsity_pred is not None:
            active_mask = self._sparsity_pred.predict_active(gate)
            gate        = gate * active_mask[None, :]
            self._sparsity_pred.update(gate)
        elif x.shape[0] == 1:
            # Even without predictor, apply a threshold for single-token
            # decode (most common path) to get cheap sparsity benefit.
            gate = gate * (np.abs(gate) >= 0.01)

        up  = self.W_u.forward(x)           # (seq, d_ff)
        return self.W_d.forward(gate * up)  # (seq, d_model)

    def _numpy_forward(self, x: np.ndarray) -> np.ndarray:
        """Standard NumPy SwiGLU forward."""
        gate = mm(x, self.W_g)
        gate = silu(gate)
        return mm(gate * mm(x, self.W_u), self.W_d)

    # ── Symmetry breaking (for MoE upcycling) ────────────────────────────────

    def _break_symmetry(self, scale: float = 1e-3):
        """Add small noise to float weights to break expert symmetry."""
        from ..quant.ternary_linear import TernaryLinear
        for attr in ("W_g", "W_u", "W_d"):
            W = getattr(self, attr)
            if isinstance(W, np.ndarray):
                W += np.random.randn(*W.shape).astype(np.float32) * scale
            # TernaryLinear / QuantizedLinear: skip (already have variance)