# DracoAI V1 — modeling/layers/moe.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Mixture-of-Experts Layer.
Routing, dispatch, stacked-weight fast path, load-balance adaptation.

NEW in this revision:
  ✅ FEAT-Z-LOSS                  : Router Z-loss reported in aux dict.
     Z-loss = mean(log(sum(exp(logits)))²) — penalises large router
     logits during training, producing smoother routing distributions
     and reducing expert collapse without changing inference routing.
     Reported as aux["z_loss"] for external training loops; does NOT
     change the routing decision (top-k unchanged).
  ✅ FEAT-ROUTER-SMOOTHING        : Softmax-temperature parameter
     (router_temp) softens the top-k gate weights.  Default 1.0 =
     original behaviour.  Values < 1 sharpen routing; > 1 distributes
     load across more experts per token.
  ✅ FEAT-EXPERT-CHOICE-TRACKING  : Reports per-expert capacity
     utilisation in aux["capacity_util"].  Expert Choice routing
     (Expert picks tokens) is the training-phase load balancer; at
     inference we keep standard top-k but track whether any expert
     exceeds its token capacity so callers can monitor collapse.
  ✅ FEAT-TERNARY-FAST-PATH       : When experts are TernaryLinear,
     the stacked fast path checks isinstance(TernaryLinear) and calls
     their addition-only forward instead of einsum — no float muls.

FIXES retained from prior revision:
  ✅ FIX-LB-RESET-GUARD           : adapt_router_bias() resets counts
     even when _lb_steps==0.
  ✅ FIX-INTENT-BIAS-SHAPE        : validated against n_experts.
  ✅ FIX-ROUTER-DTYPE             : W_router kept float32.
  ✅ FIX-STACKED-QUANT-CHECK      : checks W_g, W_u, W_d for quant.
  ✅ FIX-UNUSED-IMPORT-MM         : removed unused mm import.
  ✅ FIX-LB-EARLY-RETURN-NO-RESET : reset before early return.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from .mlp              import ExpertFFN
from ..ops.activation   import silu
from ..constants        import MOE_NOISE_SCALE, MOE_TOP_K
from ..quant.int4       import QuantizedLinear

__all__ = ["MoELayer"]

logger = logging.getLogger(__name__)


class MoELayer:
    def __init__(
        self,
        d_model:     int,
        d_ff:        int,
        n_experts:   int   = 8,
        top_k:       int   = MOE_TOP_K,
        router_temp: float = 1.0,
        ternary_experts: bool = False,
    ):
        """
        Parameters
        ----------
        router_temp     : Gate softmax temperature (1.0 = default).
                          Lower = sharper routing, higher = smoother load.
        ternary_experts : If True, convert all expert FFN weights to
                          TernaryLinear at init (addition-only forward).
        """
        self.d_model        = d_model
        self.d_ff           = d_ff
        self.n_experts      = n_experts
        self.top_k          = top_k
        self.router_temp    = max(1e-3, float(router_temp))

        scale            = 1.0 / (d_model ** 0.5)
        self.W_router    = np.random.randn(d_model, n_experts).astype(np.float32) * scale
        self.router_bias = np.zeros(n_experts, dtype=np.float32)
        self.experts     = [
            ExpertFFN(d_model, d_ff, ternary=ternary_experts)
            for _ in range(n_experts)
        ]
        self.shared = ExpertFFN(d_model, d_ff, ternary=ternary_experts)

        self._expert_counts = np.zeros(n_experts, dtype=np.int64)
        self._lb_steps      = 0
        self._stacked_valid = False
        self._W_g_stk: Optional[np.ndarray] = None
        self._W_u_stk: Optional[np.ndarray] = None
        self._W_d_stk: Optional[np.ndarray] = None

        # Expert Choice capacity tracking (inference monitoring only)
        self._capacity_factor: float = 1.25

    # ── Ternary conversion ────────────────────────────────────────────────────

    def ternarize_experts(self) -> "MoELayer":
        """Convert all expert (and shared) FFN weights to TernaryLinear."""
        for exp in list(self.experts) + [self.shared]:
            exp.ternarize()
        self._invalidate_stacked()
        return self

    # ── Stacked weights fast path ─────────────────────────────────────────────

    def _get_stacked_weights(self) -> Tuple[Optional[np.ndarray], ...]:
        if self._stacked_valid:
            return self._W_g_stk, self._W_u_stk, self._W_d_stk

        # ✅ FIX-STACKED-QUANT-CHECK: bail if ANY weight is quantized or ternary
        # (ternary has its own forward; stacked einsum would fail on non-ndarray)
        try:
            from ..quant.ternary_linear import TernaryLinear
            _ternary_cls: tuple = (QuantizedLinear, TernaryLinear)
        except ImportError:
            _ternary_cls = (QuantizedLinear,)

        for e in self.experts:
            if (isinstance(e.W_g, _ternary_cls)
                    or isinstance(e.W_u, _ternary_cls)
                    or isinstance(e.W_d, _ternary_cls)):
                return None, None, None

        self._W_g_stk = np.stack([e.W_g for e in self.experts], axis=0)
        self._W_u_stk = np.stack([e.W_u for e in self.experts], axis=0)
        self._W_d_stk = np.stack([e.W_d for e in self.experts], axis=0)
        self._stacked_valid = True
        return self._W_g_stk, self._W_u_stk, self._W_d_stk

    def _invalidate_stacked(self):
        self._stacked_valid = False
        self._W_g_stk = self._W_u_stk = self._W_d_stk = None

    # ── Load balance ──────────────────────────────────────────────────────────

    def adapt_router_bias(
        self,
        imbalance_thresh: float = 0.3,
        correction_scale: float = 0.1,
        reset_counts:     bool  = True,
    ):
        """
        Adjust router bias to reduce expert load imbalance.

        ✅ FIX-LB-RESET-GUARD + FIX-LB-EARLY-RETURN-NO-RESET: reset
        counts before returning when _lb_steps == 0.
        """
        if self._lb_steps == 0:
            if reset_counts:
                self._expert_counts[:] = 0
            return

        total = self._expert_counts.sum()
        if total > 0:
            fracs     = self._expert_counts / total
            ideal     = 1.0 / self.n_experts
            deviation = fracs - ideal
            adj  = -np.clip(deviation / (ideal + 1e-9), -1.0, 1.0) * correction_scale
            mask = np.abs(deviation) > ideal * imbalance_thresh
            self.router_bias[mask] += adj[mask].astype(np.float32)

        if reset_counts:
            self._expert_counts[:] = 0
            self._lb_steps         = 0

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:            np.ndarray,
        add_noise:    bool              = True,
        intent_bias:  Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        MoE forward pass.

        Returns (output, aux) where aux contains:
          importance_loss : std of per-expert importance (training signal)
          load_loss       : std of per-expert load fraction (training signal)
          aux_total       : sum of importance_loss + load_loss
          z_loss          : router Z-loss (training stability signal) ← NEW
          capacity_util   : per-expert capacity utilisation array    ← NEW
        """
        bsz, seq, d = x.shape
        x_flat = x.reshape(seq, d).astype(np.float32)

        # ── Router (always FP32 — RED zone) ──────────────────────────
        W_router_f32 = (self.W_router.astype(np.float32)
                        if self.W_router.dtype != np.float32 else self.W_router)
        logits = x_flat @ W_router_f32 + self.router_bias  # (seq, n_experts)

        # ── Intent bias ───────────────────────────────────────────────
        if intent_bias is not None:
            ib = np.asarray(intent_bias, dtype=np.float32).ravel()
            if ib.shape[0] == self.n_experts:
                logits = logits + ib[None, :]
            else:
                logger.warning(
                    "[MoE] intent_bias shape %s != n_experts %d — ignored",
                    intent_bias.shape, self.n_experts)

        # ── Z-loss (NEW: reported for training loops, no effect on routing) ──
        # Z-loss = mean(log(sum(exp(logits)))²)
        # Stabilised log-sum-exp to avoid overflow
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        log_z = np.log(
            np.exp(np.clip(logits_shifted, -50, 50)).sum(axis=-1) + 1e-9
        ) + logits.max(axis=-1)
        z_loss = float((log_z ** 2).mean())

        # ── Routing soft weights (for aux loss) ──────────────────────
        router_soft = np.exp(np.clip(logits_shifted, -50, 50))
        router_soft = router_soft / (router_soft.sum(axis=-1, keepdims=True) + 1e-9)

        # ── Gumbel noise for training diversity ──────────────────────
        if add_noise and seq > 0:
            noise = (np.random.gumbel(size=logits.shape).astype(np.float64)
                     * MOE_NOISE_SCALE).astype(np.float32)
            logits = logits + noise

        # ── Top-K routing ─────────────────────────────────────────────
        top_idx    = np.argsort(logits, axis=-1)[:, -self.top_k:][:, ::-1]
        top_logits = np.take_along_axis(logits, top_idx, axis=1)
        # ✅ FEAT-ROUTER-SMOOTHING: apply temperature before softmax gate
        top_logits = top_logits / self.router_temp
        top_logits = top_logits - top_logits.max(axis=-1, keepdims=True)
        gates      = np.exp(np.clip(top_logits, -50, 50))
        gates      = gates / (gates.sum(axis=-1, keepdims=True) + 1e-9)

        # ── Expert count tracking ─────────────────────────────────────
        unique_eids, unique_counts = np.unique(top_idx, return_counts=True)
        for eid, cnt in zip(unique_eids, unique_counts):
            self._expert_counts[int(eid)] += int(cnt)
        self._lb_steps += 1

        # ── Expert Choice capacity monitoring ─────────────────────────
        # (inference-only monitoring; does not change routing)
        # capacity = ceil(seq * top_k / n_experts * capacity_factor)
        capacity = max(1, int(
            math.ceil(seq * self.top_k / self.n_experts * self._capacity_factor)
        ))
        # Count actual tokens per expert
        expert_token_counts = np.zeros(self.n_experts, dtype=np.int32)
        for eid, cnt in zip(unique_eids, unique_counts):
            expert_token_counts[int(eid)] = int(cnt)
        capacity_util = expert_token_counts / max(1, capacity)

        output = np.zeros((seq, d), dtype=np.float32)

        # ── Stacked einsum fast path (float experts only) ─────────────
        W_g_stk, W_u_stk, W_d_stk = self._get_stacked_weights()
        if W_g_stk is not None and seq > 0:
            for k in range(self.top_k):
                expert_ids = top_idx[:, k]
                g_k        = gates[:, k]
                W_g_sel    = W_g_stk[expert_ids]  # (seq, d_model, d_ff)
                W_u_sel    = W_u_stk[expert_ids]
                W_d_sel    = W_d_stk[expert_ids]
                gate_act   = silu(np.einsum("bi,bij->bj", x_flat, W_g_sel))
                up_act     = np.einsum("bi,bij->bj", x_flat, W_u_sel)
                down_out   = np.einsum("bi,bij->bj", gate_act * up_act, W_d_sel)
                output    += g_k[:, None] * down_out

        # ── Per-expert dispatch (ternary / quantized experts) ─────────
        elif seq > 0:
            for k in range(self.top_k):
                expert_ids = top_idx[:, k]
                g_k        = gates[:, k]
                unique_experts, inverse = np.unique(expert_ids, return_inverse=True)
                for local_e, e in enumerate(unique_experts):
                    mask  = inverse == local_e
                    x_sel = x_flat[mask]
                    g_sel = g_k[mask]
                    e_out = self.experts[e].forward(x_sel)
                    output[mask] += g_sel[:, None] * e_out

        # ── Shared expert (always active — DeepSeek style) ────────────
        output += self.shared.forward(x_flat)

        # ── Auxiliary losses ──────────────────────────────────────────
        importance = router_soft.mean(axis=0)
        load = (top_idx[None, :, :] == np.arange(self.n_experts)[:, None, None]
                ).any(axis=-1).mean(axis=1)
        aux = dict(
            importance_loss = float(importance.std()),
            load_loss       = float(load.std()),
            aux_total       = float(importance.std() + load.std()),
            z_loss          = z_loss,
            capacity_util   = capacity_util,
        )
        return output.reshape(bsz, seq, d), aux


# ── Missing math import ───────────────────────────────────────────────────────
import math