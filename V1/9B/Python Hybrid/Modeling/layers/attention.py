# DracoAI V1 — modeling/layers/attention.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Grouped Query Attention (GQA) — with deep Engram integration.

Kernel dispatch order:
  1. Triton fused_attention (GPU, causal)
  2. NumPy (always available)

Engram cross-attention (deep integration):
  When an EngramCache is attached (via block.py forwarding), the output is a
  blended combination of:
    • exact sliding-window attention  (weight = eff_alpha)
    • engram cross-attention          (weight = 1 - eff_alpha)

  eff_alpha is returned by engram.attend() and may be dynamically reduced
  when the engram has high-confidence matches for the current query.
  The blend happens BEFORE the output projection W_o, so engram context
  flows through the same projection as exact context.

FIXES (this revision):
  ✅ FIX-ENGRAM-DEEP-INTEGRATION   : Engram cross-attention blended at the
     hidden-state level before W_o projection.
  ✅ FIX-DYNAMIC-ALPHA-API         : attend() returns (eng_out, eff_alpha).
  ✅ FIX-ENCAPSULATION-CACHE-POS   : cache.get_pos() used instead of
     accessing _cache_pos directly.
  ✅ FIX-ROPE-THETA-CACHE          : _get_rope() now tracks _rope_theta_cached
     alongside head_dim so that a post-construction change of _rope_theta
     (e.g. PrefixCache restore with different theta) correctly invalidates
     and recomputes the frequency table.  Without this fix, stale RoPE
     frequencies would silently corrupt positional encodings.
"""
from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING
import numpy as np

from ..ops.attention_ops import rope_freqs as _rope_freqs, apply_rope
from ..ops.tensor_ops    import mm
from ..constants         import ROPE_THETA, LOGIT_CLIP, SOFTMAX_EPS

if TYPE_CHECKING:
    from ..kv_cache.kv_cache      import KVCache
    from ..kv_cache.engram_cache  import EngramCache
    from ..runtime.tensor_pool    import TensorPool

__all__ = ["GQAttention"]


class GQAttention:
    """
    Grouped Query Attention with optional deep Engram integration.
    n_rep = n_heads // n_kv_heads (MHA when n_rep == 1).
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, rope_theta: float = ROPE_THETA):
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = head_dim
        self.n_rep       = n_heads // n_kv_heads
        self._rope_theta = rope_theta

        scale = 1.0 / math.sqrt(d_model)
        self.W_q = np.random.randn(d_model, n_heads    * head_dim).astype(np.float32) * scale
        self.W_k = np.random.randn(d_model, n_kv_heads * head_dim).astype(np.float32) * scale
        self.W_v = np.random.randn(d_model, n_kv_heads * head_dim).astype(np.float32) * scale
        self.W_o = np.random.randn(n_heads * head_dim, d_model).astype(np.float32)    * scale

        # ✅ FIX-ROPE-THETA-CACHE: track both head_dim and rope_theta so that
        # a post-construction change of either attribute correctly invalidates
        # the cached frequency table.
        self._rope_freqs_cache:  Optional[np.ndarray] = None
        self._rope_theta_cached: float = rope_theta          # <-- NEW
        self._causal_mask:       Optional[np.ndarray] = None
        self._causal_mask_size:  int = 0

    def _get_rope(self) -> np.ndarray:
        """Return RoPE frequency table, recomputing if head_dim or rope_theta changed."""
        # ✅ FIX-ROPE-THETA-CACHE: invalidate when theta changes, not just head_dim.
        if (
            self._rope_freqs_cache is None
            or self._rope_freqs_cache.shape[0] != self.head_dim // 2
            or self._rope_theta_cached != self._rope_theta   # <-- NEW check
        ):
            self._rope_freqs_cache  = _rope_freqs(self.head_dim, self._rope_theta)
            self._rope_theta_cached = self._rope_theta
        return self._rope_freqs_cache

    def forward(
        self,
        x:          np.ndarray,          # (1, seq, d_model)
        cache:      "KVCache",
        layer_idx:  int,
        snap:       Optional[dict]         = None,
        batch_idx:  int                    = 0,
        pool:       Optional["TensorPool"] = None,
        engram:     Optional["EngramCache"] = None,
    ) -> np.ndarray:
        """
        x: (1, seq, d_model) → (1, seq, d_model)

        When engram is provided and has committed blocks:
          eng_out, eff_alpha = engram.attend(...)
          out = eff_alpha * exact_out + (1 - eff_alpha) * eng_out

        eff_alpha may be < blend_alpha when engram has high-confidence
        matches for the current query (dynamic blend alpha feature).
        """
        bsz, seq, _ = x.shape
        freqs  = self._get_rope()
        offset = cache.get_pos(batch_idx)   # ✅ public accessor

        Q = mm(x, self.W_q).reshape(bsz, seq, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        K = mm(x, self.W_k).reshape(bsz, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = mm(x, self.W_v).reshape(bsz, seq, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        Q = apply_rope(Q, freqs, offset)
        K = apply_rope(K, freqs, offset)

        cache.update(layer_idx, K, V, snap=snap, batch_idx=batch_idx)
        K_f, V_f = cache.get(layer_idx, batch_idx=batch_idx)
        kv_seq   = K_f.shape[2]

        K_exp = np.repeat(K_f, self.n_rep, axis=1)
        V_exp = np.repeat(V_f, self.n_rep, axis=1)

        scale_ = 1.0 / math.sqrt(self.head_dim)

        # ── Triton fused attention (GPU) ──────────────────────────────
        _triton_out = None
        try:
            from ..kernels import get_kernel
            kern = get_kernel("fused_attention")
            if kern is not None:
                _triton_out = kern(
                    Q.astype(np.float32), K_exp.astype(np.float32),
                    V_exp.astype(np.float32), scale_, causal=(seq > 1))
        except Exception:
            pass

        if _triton_out is not None:
            out = _triton_out
        else:
            # ── NumPy path ────────────────────────────────────────────
            attn_shape = (bsz, self.n_heads, seq, kv_seq)
            if pool is not None:
                attn = pool.get(attn_shape, np.float32)
                np.multiply(Q @ K_exp.transpose(0, 1, 3, 2), scale_, out=attn)
            else:
                attn = Q @ K_exp.transpose(0, 1, 3, 2) * scale_

            if seq > 1:
                if self._causal_mask is None or self._causal_mask_size < seq:
                    new_size = max(seq, self._causal_mask_size * 2
                                   if self._causal_mask_size else seq)
                    self._causal_mask = np.triu(
                        np.full((new_size, new_size), -1e9, dtype=np.float32), 1)
                    self._causal_mask_size = new_size
                causal   = self._causal_mask[:seq, :seq]
                past_len = kv_seq - seq
                mask_full = (np.concatenate(
                    [np.zeros((seq, past_len), dtype=np.float32), causal], axis=1)
                    if past_len > 0 else causal)
                attn = attn + mask_full[None, None, :, :]

            attn = np.clip(attn, -LOGIT_CLIP, LOGIT_CLIP)
            attn = attn - attn.max(axis=-1, keepdims=True)
            attn = np.exp(attn)
            attn = attn / (attn.sum(axis=-1, keepdims=True) + SOFTMAX_EPS)
            out  = attn @ V_exp
            if pool is not None:
                pool.put(attn)

        # ── Deep Engram cross-attention blend ─────────────────────────
        if engram is not None and len(engram) > 0:
            eng_out, eff_alpha = engram.attend(
                layer_idx   = layer_idx,
                Q           = Q,
                n_rep       = self.n_rep,
                scale       = scale_,
                softmax_eps = SOFTMAX_EPS,
            )
            if eng_out is not None:
                out = eff_alpha * out + (1.0 - eff_alpha) * eng_out

        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq, self.n_heads * self.head_dim)
        return mm(out, self.W_o)