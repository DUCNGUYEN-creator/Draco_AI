# DracoAI V1 — modeling/layers/moe.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Mixture-of-Experts Layer.
Routing, dispatch, stacked-weight fast path, load-balance adaptation.

FIXES (this revision):
    ✅ FIX-LB-RESET-GUARD      : adapt_router_bias() now resets _expert_counts
       and _lb_steps even when total==0 to prevent unbounded accumulation of
       zero-count steps that would corrupt future imbalance measurements.
    ✅ FIX-INTENT-BIAS-SHAPE   : intent_bias is validated against n_experts before
       being applied to router logits.  Previously a mis-sized bias silently
       broadcast incorrectly.
    ✅ FIX-ROUTER-DTYPE        : W_router matmul explicitly kept in float32 to
       avoid silent precision loss when model is cast to float16.
    ✅ FIX-STACKED-QUANT-CHECK : _get_stacked_weights() now checks ALL three
       expert weight matrices (W_g, W_u, W_d) for QuantizedLinear, not only
       W_g.  Previously, if W_g was a plain ndarray but W_u or W_d was a
       QuantizedLinear, np.stack() would produce an object-dtype array and the
       subsequent einsum would raise a TypeError at runtime after quantization.
    ✅ FIX-UNUSED-IMPORT-MM    : removed unused `mm` import from ops.tensor_ops.
    ✅ FIX-LB-EARLY-RETURN-NO-RESET : adapt_router_bias() previously returned
       immediately when _lb_steps == 0 WITHOUT resetting _expert_counts.
       This violated the documented FIX-LB-RESET-GUARD invariant: if
       reset_counts=True is passed and _lb_steps happens to be 0 (e.g. after
       a manual external reset of only _lb_steps), _expert_counts would keep
       stale values indefinitely.  Now the reset is performed BEFORE the early
       return so the invariant "reset_counts=True always clears counts" holds
       unconditionally regardless of _lb_steps.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from .mlp             import ExpertFFN
from ..ops.activation  import silu
from ..constants       import MOE_NOISE_SCALE, MOE_TOP_K
from ..quant.int4      import QuantizedLinear

__all__ = ["MoELayer"]

logger = logging.getLogger(__name__)


class MoELayer:
    def __init__(self, d_model: int, d_ff: int,
                 n_experts: int = 8, top_k: int = MOE_TOP_K):
        self.d_model   = d_model
        self.d_ff      = d_ff
        self.n_experts = n_experts
        self.top_k     = top_k

        scale            = 1.0 / (d_model ** 0.5)
        self.W_router    = np.random.randn(d_model, n_experts).astype(np.float32) * scale
        self.router_bias = np.zeros(n_experts, dtype=np.float32)
        self.experts     = [ExpertFFN(d_model, d_ff) for _ in range(n_experts)]
        self.shared      = ExpertFFN(d_model, d_ff)

        self._expert_counts = np.zeros(n_experts, dtype=np.int64)
        self._lb_steps      = 0
        self._stacked_valid = False
        self._W_g_stk: Optional[np.ndarray] = None
        self._W_u_stk: Optional[np.ndarray] = None
        self._W_d_stk: Optional[np.ndarray] = None

    def _get_stacked_weights(self) -> Tuple[Optional[np.ndarray], ...]:
        if self._stacked_valid:
            return self._W_g_stk, self._W_u_stk, self._W_d_stk

        # ✅ FIX-STACKED-QUANT-CHECK: check W_g, W_u, and W_d for all experts.
        # np.stack() on QuantizedLinear objects produces an object-dtype array;
        # the einsum fast path then crashes with a TypeError.  We must bail out
        # to the per-expert slow path whenever ANY weight is quantized.
        for e in self.experts:
            if (isinstance(e.W_g, QuantizedLinear)
                    or isinstance(e.W_u, QuantizedLinear)
                    or isinstance(e.W_d, QuantizedLinear)):
                return None, None, None

        self._W_g_stk = np.stack([e.W_g for e in self.experts], axis=0)
        self._W_u_stk = np.stack([e.W_u for e in self.experts], axis=0)
        self._W_d_stk = np.stack([e.W_d for e in self.experts], axis=0)
        self._stacked_valid = True
        return self._W_g_stk, self._W_u_stk, self._W_d_stk

    def _invalidate_stacked(self):
        self._stacked_valid = False
        self._W_g_stk = self._W_u_stk = self._W_d_stk = None

    def adapt_router_bias(self, imbalance_thresh: float = 0.3,
                          correction_scale: float = 0.1, reset_counts: bool = True):
        """Adjust router bias to reduce expert load imbalance.

        ✅ FIX-LB-RESET-GUARD + FIX-LB-EARLY-RETURN-NO-RESET:
        When _lb_steps == 0 (no forward passes since last reset), there is
        nothing to adjust — but if reset_counts=True, _expert_counts must
        still be cleared to satisfy the invariant "reset_counts=True always
        clears stale counts regardless of _lb_steps".

        Previously the early-return fired BEFORE the reset block at the end
        of the function, silently leaving stale count data in _expert_counts
        when _lb_steps happened to be 0.

        Fix: the _lb_steps == 0 early-return now performs the reset inline
        before returning, so the invariant holds unconditionally.  For the
        normal path (_lb_steps > 0), the adjustment runs first and the reset
        follows — preserving the intended adjust-then-reset semantics.
        """
        # ✅ FIX-LB-EARLY-RETURN-NO-RESET: nothing to adjust when _lb_steps==0,
        # but still honour the reset contract before returning early.
        if self._lb_steps == 0:
            if reset_counts:
                self._expert_counts[:] = 0
                # _lb_steps is already 0, nothing more to do
            return

        total = self._expert_counts.sum()
        if total > 0:
            fracs     = self._expert_counts / total
            ideal     = 1.0 / self.n_experts
            deviation = fracs - ideal
            adj  = -np.clip(deviation / (ideal + 1e-9), -1.0, 1.0) * correction_scale
            mask = np.abs(deviation) > ideal * imbalance_thresh
            self.router_bias[mask] += adj[mask].astype(np.float32)

        # Always reset regardless of total — prevents stale count accumulation.
        if reset_counts:
            self._expert_counts[:] = 0
            self._lb_steps         = 0

    def forward(self, x: np.ndarray, add_noise: bool = True,
                intent_bias: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        bsz, seq, d = x.shape
        x_flat = x.reshape(seq, d).astype(np.float32)

        # Keep router in float32 even when model is float16
        W_router_f32 = (self.W_router.astype(np.float32)
                        if self.W_router.dtype != np.float32 else self.W_router)
        logits = x_flat @ W_router_f32 + self.router_bias

        # Validate and apply intent bias
        if intent_bias is not None:
            ib = np.asarray(intent_bias, dtype=np.float32).ravel()
            if ib.shape[0] == self.n_experts:
                logits = logits + ib[None, :]
            else:
                logger.warning(
                    "[MoE] intent_bias shape %s != n_experts %d — ignored",
                    intent_bias.shape, self.n_experts)

        router_soft = np.exp(np.clip(logits - logits.max(axis=-1, keepdims=True), -50, 50))
        router_soft = router_soft / (router_soft.sum(axis=-1, keepdims=True) + 1e-9)

        if add_noise and seq > 0:
            noise = (np.random.gumbel(size=logits.shape).astype(np.float64)
                     * MOE_NOISE_SCALE).astype(np.float32)
            logits = logits + noise

        top_idx    = np.argsort(logits, axis=-1)[:, -self.top_k:][:, ::-1]
        top_logits = np.take_along_axis(logits, top_idx, axis=1)
        top_logits = top_logits - top_logits.max(axis=-1, keepdims=True)
        gates      = np.exp(np.clip(top_logits, -50, 50))
        gates      = gates / (gates.sum(axis=-1, keepdims=True) + 1e-9)

        output = np.zeros((seq, d), dtype=np.float32)

        unique_eids, unique_counts = np.unique(top_idx, return_counts=True)
        for eid, cnt in zip(unique_eids, unique_counts):
            self._expert_counts[int(eid)] += int(cnt)
        self._lb_steps += 1

        W_g_stk, W_u_stk, W_d_stk = self._get_stacked_weights()
        if W_g_stk is not None and seq > 0:
            # Batched einsum fast path (no per-expert Python loops)
            for k in range(self.top_k):
                expert_ids = top_idx[:, k]
                g_k        = gates[:, k]
                W_g_sel    = W_g_stk[expert_ids]
                W_u_sel    = W_u_stk[expert_ids]
                W_d_sel    = W_d_stk[expert_ids]
                gate_act   = silu(np.einsum("bi,bij->bj", x_flat, W_g_sel))
                up_act     = np.einsum("bi,bij->bj", x_flat, W_u_sel)
                down_out   = np.einsum("bi,bij->bj", gate_act * up_act, W_d_sel)
                output    += g_k[:, None] * down_out
        else:
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

        output += self.shared.forward(x_flat)

        # Auxiliary losses for MoE load balance monitoring
        importance = router_soft.mean(axis=0)
        load = (top_idx[None, :, :] == np.arange(self.n_experts)[:, None, None]
                ).any(axis=-1).mean(axis=1)
        aux = dict(importance_loss=float(importance.std()),
                   load_loss=float(load.std()),
                   aux_total=float(importance.std() + load.std()))
        return output.reshape(bsz, seq, d), aux