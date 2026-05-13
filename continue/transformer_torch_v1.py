# Copyright 2026 Draco Studio and DUCNGUYEN-creator
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DracoAI V1 — PyTorch Transformer (Training)
============================================
GPU-accelerated training counterpart to transformer_v1.py.
Uses Flash Attention (F.scaled_dot_product_attention), AMP, gradient clipping.

FIXES (V1 — final consolidated):
    ✅ RoPE: correct offset passed from cache_pos (not hardcoded 0)
    ✅ MoE: vectorised boolean-mask dispatch
    ✅ MTP: weight-tied lm_head (single parameter, not duplicated)
    ✅ _full_sync: copies ALL parameters to NumPy inference model
    ✅ router_bias: separate nn.Parameter (not Linear bias)
    ✅ _break_symmetry: NOT called in __init__
    ✅ load_external_weights: expert weight AVERAGING
    ✅ bias=False on all Linear layers (matches Qwen 3.5 9B "NO BIAS" spec)
    ✅ KVCache: long-prefill safe path
    ✅ FIX M2: aux_total accumulated and returned from forward()
    ✅ FIX: is_causal = (seq > 1)
    ✅ FIX: KVCacheTorch.get() rec_start formula matches NumPy version
    ✅ FIX D2: set_identity_bias() default boost 5.0 → 2.0
    ✅ FIX 3.1: _full_sync validates tensor shapes before every assignment
    ✅ FIX 3.2: MoELayerTorch.forward() wraps aux_loss in nan_to_num
    ✅ FIX 3.3: load_checkpoint() config compatibility check
    ✅ FIX MOE-NOISE-TRAIN: Gumbel noise in MoE router at inference time
    ✅ FIX ATTN-CLIP-TORCH: attention scores clipped in manual fallback path
    ✅ FIX LOGITS-STABILITY: forward() clips l1/l2 output to [-100, 100]
    ✅ FIX LORA: LoRALinear class for parameter-efficient fine-tuning
    ✅ FIX GRADIENT-CLIP: train_step uses clip_grad_norm_ = 1.0
    ✅ FIX NAN-GUARD: train_step catches NaN loss and skips step
    ✅ FIX CHECKPOINT-LATEST: load_checkpoint() sorts by mtime not lex
    ✅ FIX TRAIN-STEP (🔴 CRITICAL): Removed torch.no_grad().__class__()
         syntax error. Was causing AttributeError on every training call.
         Replaced with clean AMP/no-grad branching pattern. The old code had:
             with (torch.cuda.amp.autocast() if use_amp else torch.no_grad().__class__()):
         which raises AttributeError because no_grad() is a context manager
         instance, not a class. Now uses a proper contextlib.nullcontext()
         fallback for the non-AMP path, which is the correct Python idiom.
    ✅ FIX KVCACHE-ALLOC: KVCacheTorch uses torch.empty instead of torch.zeros.
         torch.zeros zero-fills the entire buffer at init time (~2 GB/buffer for 9B model).
         torch.empty skips zero-fill; safe because update() always writes before get() reads.
         This eliminates startup lag and prevents OOM on low-VRAM / low-RAM machines.
    ✅ FIX KVCACHE-LONGPREFILL-POS (🔴 CRITICAL): KVCacheTorch long-prefill path (seq > window)
         left cache_pos=0. Then step(seq) pushed cache_pos >> window causing get() rec_start
         to point to the middle of the tail buffer — wrong token order on first decode step
         after a long prompt. Fix: long-prefill sets cache_pos=window; step() guards double-advance.
    ✅ FIX KVCACHE-SNAPSHOT: KVCacheTorch gains snapshot()/restore() — parity with NumPy KVCache,
         enabling transactional speculative decoding rollback in Torch inference.
    ✅ FIX MTP-TRAIN-SIGNAL: train_step now adds auxiliary CE loss on l2 (next+1 prediction,
         weight mtp_aux_coeff=0.1). Previously W1/W2 received no direct training signal.
    ✅ FIX DDP-VOCAB: train_step uses model.module when DDP-wrapped (vocab_size access safe).
    ✅ FIX TORCH-LOAD: torch.load() passes weights_only=False to suppress PyTorch 2.4+ warning.
    ✅ FIX FREEZE-EXPERTS: Added freeze_experts()/unfreeze_experts() for router-only training.
    ✅ QLORA: enable_qlora(rank, alpha) replaces attention projections with bitsandbytes NF4
         4-bit quantised base + float16 LoRA adapters. Enables full-model fine-tuning on a
         single 24 GB GPU for a 9B model. merge_qlora() dequantises and absorbs adapters
         before _full_sync() to the NumPy inference model.
    ✅ EXPORT GGUF: export_gguf(path) creates a temporary NumPy model, syncs weights, then
         calls GGUFExporter.write_gguf(). After export, quantise with llama.cpp:
         llama-quantize out_fp16.gguf out_q4km.gguf Q4_K_M"""

from __future__ import annotations

import math, os, json
from contextlib import nullcontext
from typing import List, Optional, Tuple, Dict

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────

def _rope_freqs_torch(head_dim: int, base: float = 10000.0,
                      device=None) -> "torch.Tensor":
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (base ** (i / head_dim))

def _apply_rope_torch(x: "torch.Tensor", freqs: "torch.Tensor",
                      offset: int = 0) -> "torch.Tensor":
    """Apply RoPE to x of shape (batch, heads, seq, head_dim)."""
    seq  = x.shape[2]
    hdim = x.shape[3]
    pos  = torch.arange(offset, offset + seq, dtype=torch.float32, device=x.device)
    angles = torch.outer(pos, freqs)
    cos    = angles.cos()[None, None, :, :]
    sin    = angles.sin()[None, None, :, :]
    x1, x2 = x[..., :hdim // 2], x[..., hdim // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ─────────────────────────────────────────────────────────────────────
# LoRA Linear Layer (for parameter-efficient fine-tuning)
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class LoRALinear(nn.Module):
        """
        FIX LORA: Low-Rank Adaptation wrapper for nn.Linear.
        Wraps W_q/W_v (or any projection) for PEFT.

        Usage:
            layers = LoRALinear(in_features, out_features, rank=8, alpha=16.0)
            # During training: only layers.lora_A, layers.lora_B are updated.
            # Freeze base: layers.linear.requires_grad_(False)
            # After training: layers.merge() then pass to _full_sync().

        The effective weight is: W + (alpha/rank) * lora_B @ lora_A
        """

        def __init__(self, in_features: int, out_features: int,
                     rank: int = 8, alpha: float = 16.0, bias: bool = False):
            super().__init__()
            self.linear  = nn.Linear(in_features, out_features, bias=bias)
            self.lora_A  = nn.Parameter(torch.randn(rank, in_features) * 0.02)
            self.lora_B  = nn.Parameter(torch.zeros(out_features, rank))
            self.alpha   = alpha
            self.rank    = rank
            self.merged  = False
            self.scaling = alpha / rank

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base = self.linear(x)
            if self.merged:
                return base
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            return base + lora_out * self.scaling

        def merge(self):
            """Merge LoRA weights into the base linear weight (irreversible)."""
            if self.merged:
                return
            with torch.no_grad():
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.linear.weight.data += delta
            self.merged = True

        def unmerge(self):
            """Unmerge LoRA weights (restore base weight)."""
            if not self.merged:
                return
            with torch.no_grad():
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.linear.weight.data -= delta
            self.merged = False

# ─────────────────────────────────────────────────────────────────────
# KVCache (PyTorch)
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class KVCacheTorch:
        """
        Torch KVCache mirroring NumPy KVCache.
        FIX: prefill path uses sink+tail copy, no modulo corruption.
        FIX: get() rec_start formula matches NumPy version exactly.
        """

        def __init__(self, n_layers, n_kv_heads, head_dim,
                     window=1024, sink=4, device=None):
            self.window    = window
            self.sink      = sink
            self.n_layers  = n_layers
            self.cache_pos = 0
            self.filled    = 0
            shape = (n_layers, 1, n_kv_heads, window, head_dim)
            # FIX KVCACHE-ALLOC: torch.empty skips zero-fill — fast startup.
            # Safe because update() always writes a slot before get() reads it.
            self.k_buf = torch.empty(shape, dtype=torch.float16, device=device)
            self.v_buf = torch.empty(shape, dtype=torch.float16, device=device)

        def reset(self):
            self.k_buf.zero_()
            self.v_buf.zero_()
            self.cache_pos = 0
            self.filled    = 0

        def update(self, layer, k, v):
            seq = k.shape[2]
            if seq > self.window:
                tail_len = self.window - self.sink
                self.k_buf[layer, :, :, :self.sink,            :] = k[:, :, :self.sink,  :].half()
                self.v_buf[layer, :, :, :self.sink,            :] = v[:, :, :self.sink,  :].half()
                self.k_buf[layer, :, :, self.sink:self.window, :] = k[:, :, -tail_len:,  :].half()
                self.v_buf[layer, :, :, self.sink:self.window, :] = v[:, :, -tail_len:,  :].half()
                if layer == 0:
                    self.filled    = self.window
                    # FIX KVCACHE-LONGPREFILL-POS: pin cache_pos to window so get()
                    # rec_start formula returns correct chronological order on first
                    # decode step after a long prompt.  Mirrors NumPy KVCache fix.
                    self.cache_pos = self.window
            else:
                for s in range(seq):
                    abs_pos = self.cache_pos + s
                    buf_pos = (abs_pos if abs_pos < self.sink else
                               self.sink + (abs_pos - self.sink) % max(1, self.window - self.sink))
                    self.k_buf[layer, :, :, buf_pos, :] = k[:, :, s, :].half()
                    self.v_buf[layer, :, :, buf_pos, :] = v[:, :, s, :].half()
                if layer == 0:
                    self.filled = min(self.cache_pos + seq, self.window)

        def get(self, layer):
            if self.filled < self.window:
                return (self.k_buf[layer, :, :, :self.filled, :].float(),
                        self.v_buf[layer, :, :, :self.filled, :].float())

            recent_cap = self.window - self.sink
            if recent_cap <= 0:
                return (self.k_buf[layer, :, :, :self.filled, :].float(),
                        self.v_buf[layer, :, :, :self.filled, :].float())

            rec_start = self.sink + (self.cache_pos - self.sink) % recent_cap
            k_sink = self.k_buf[layer, :, :, :self.sink, :]
            v_sink = self.v_buf[layer, :, :, :self.sink, :]
            k_rec  = torch.cat([self.k_buf[layer, :, :, rec_start:self.window, :],
                                 self.k_buf[layer, :, :, self.sink:rec_start,   :]], dim=2)
            v_rec  = torch.cat([self.v_buf[layer, :, :, rec_start:self.window, :],
                                 self.v_buf[layer, :, :, self.sink:rec_start,   :]], dim=2)
            return (torch.cat([k_sink, k_rec], dim=2).float(),
                    torch.cat([v_sink, v_rec], dim=2).float())

        def step(self, seq_len=1):
            """FIX KVCACHE-LONGPREFILL-POS: guard against double-advancing after
            long-prefill already pinned cache_pos to window."""
            if self.cache_pos == self.window and seq_len > self.window:
                pass  # already pinned; skip re-increment
            else:
                self.cache_pos += seq_len

        def snapshot(self) -> dict:
            """FIX KVCACHE-SNAPSHOT: Transactional rollback support — mirrors NumPy KVCache.
            Call before forwarding a speculative token; call restore(snap) if rejected."""
            return {
                "cache_pos": self.cache_pos,
                "filled":    self.filled,
                "k_buf":     self.k_buf.clone(),
                "v_buf":     self.v_buf.clone(),
            }

        def restore(self, snap: dict):
            """FIX KVCACHE-SNAPSHOT: Restore cache to state captured by snapshot()."""
            self.cache_pos = snap["cache_pos"]
            self.filled    = snap["filled"]
            self.k_buf.copy_(snap["k_buf"])
            self.v_buf.copy_(snap["v_buf"])

# ─────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class GQAttentionTorch(nn.Module):

        def __init__(self, d_model, n_heads, n_kv_heads, head_dim):
            super().__init__()
            self.n_heads    = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim   = head_dim
            self.n_rep      = n_heads // n_kv_heads
            # FIX: bias=False on ALL projections (Qwen 3.5 9B "NO BIAS")
            self.W_q = nn.Linear(d_model, n_heads    * head_dim, bias=False)
            self.W_k = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
            self.W_v = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
            self.W_o = nn.Linear(n_heads  * head_dim, d_model,   bias=False)
            self._freqs: Optional[torch.Tensor] = None

        def _get_freqs(self, device):
            if self._freqs is None:
                self._freqs = _rope_freqs_torch(self.head_dim, device=device)
            return self._freqs.to(device)

        def forward(self, x, cache: "KVCacheTorch", layer_idx: int):
            """
            FIX: offset = cache.cache_pos (not 0).
            FIX: is_causal = (seq > 1) to avoid shape mismatch on decode step.
            """
            bsz, seq, _ = x.shape
            freqs  = self._get_freqs(x.device)
            offset = cache.cache_pos

            Q = self.W_q(x).reshape(bsz, seq, self.n_heads,    self.head_dim).permute(0, 2, 1, 3)
            K = self.W_k(x).reshape(bsz, seq, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
            V = self.W_v(x).reshape(bsz, seq, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

            Q = _apply_rope_torch(Q, freqs, offset)
            K = _apply_rope_torch(K, freqs, offset)

            cache.update(layer_idx, K, V)
            K_f, V_f = cache.get(layer_idx)

            K_exp = K_f.repeat_interleave(self.n_rep, dim=1)
            V_exp = V_f.repeat_interleave(self.n_rep, dim=1)

            # Flash Attention (torch 2.0+) — handles causal mask internally
            out = F.scaled_dot_product_attention(
                Q, K_exp, V_exp,
                is_causal=(seq > 1),
                dropout_p=0.0,
            )

            out = out.permute(0, 2, 1, 3).reshape(bsz, seq, self.n_heads * self.head_dim)
            return self.W_o(out)

# ─────────────────────────────────────────────────────────────────────
# Expert FFN
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class ExpertFFNTorch(nn.Module):

        def __init__(self, d_model, d_ff):
            super().__init__()
            self.W_g = nn.Linear(d_model, d_ff, bias=False)
            self.W_u = nn.Linear(d_model, d_ff, bias=False)
            self.W_d = nn.Linear(d_ff, d_model, bias=False)

        def forward(self, x):
            return self.W_d(F.silu(self.W_g(x)) * self.W_u(x))

        def break_symmetry(self, scale=1e-3):
            """Only call for random-init models, NOT after loading checkpoint."""
            with torch.no_grad():
                self.W_g.weight.add_(torch.randn_like(self.W_g.weight) * scale)
                self.W_u.weight.add_(torch.randn_like(self.W_u.weight) * scale)

# ─────────────────────────────────────────────────────────────────────
# MoE Layer
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class MoELayerTorch(nn.Module):
        """
        FIX M2: aux_loss returned as scalar tensor, accumulated in forward().
        FIX 3.2: nan_to_num guard on aux_loss prevents gradient poisoning.
        FIX MOE-NOISE-TRAIN: add_noise kwarg controls Gumbel noise (inference only).
        """

        def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
            super().__init__()
            self.n_experts = n_experts
            self.top_k     = top_k
            self.W_router    = nn.Linear(d_model, n_experts, bias=False)
            self.router_bias = nn.Parameter(torch.zeros(n_experts))
            self.experts     = nn.ModuleList([ExpertFFNTorch(d_model, d_ff) for _ in range(n_experts)])
            self.shared      = ExpertFFNTorch(d_model, d_ff)
            self.norm        = nn.RMSNorm(d_model)

        def forward(self, x,
                    add_noise: bool = False,
                    intent_bias: Optional[torch.Tensor] = None):
            """
            add_noise=True for inference only (mirrors NumPy MoELayer).
            Training uses aux_loss for expert diversity.
            intent_bias: optional (n_experts,) tensor — engine bias added to router logits.
            """
            bsz, seq, d = x.shape
            x_flat  = x.reshape(seq, d)
            logits  = self.W_router(x_flat) + self.router_bias

            # Kết nối Engine → Router: cộng intent_bias (đã nhân với INTENT_BIAS_ALPHA ở ngoài)
            if intent_bias is not None:
                logits = logits + intent_bias.reshape(1, -1)

            # FIX MOE-NOISE-TRAIN: Gumbel noise at inference time only
            if add_noise and not self.training:
                noise  = torch.empty_like(logits).uniform_().clamp(1e-9, 1 - 1e-9)
                noise  = -torch.log(-torch.log(noise)) * 0.05
                logits = logits + noise

            top_idx    = torch.topk(logits, self.top_k, dim=-1).indices
            top_logits = logits.gather(1, top_idx)
            gates      = F.softmax(top_logits, dim=-1)

            output   = torch.zeros_like(x_flat)
            x_normed = self.norm(x_flat)

            for k in range(self.top_k):
                expert_ids = top_idx[:, k]
                g_k        = gates[:, k]
                for e in range(self.n_experts):
                    mask = expert_ids == e
                    if not mask.any():
                        continue
                    out = self.experts[e](x_normed[mask])
                    output[mask] += g_k[mask].unsqueeze(-1) * out

            output += self.shared(x_normed)

            # Load-balance loss
            router_soft = F.softmax(logits, dim=-1)
            importance  = router_soft.mean(dim=0)
            load = (top_idx.unsqueeze(-1) == torch.arange(
                        self.n_experts, device=x.device)
                   ).any(dim=1).float().mean(dim=0)

            raw_loss = importance.std() + load.std()
            # FIX 3.2: Guard against NaN from empty batch
            aux_loss = torch.nan_to_num(raw_loss, nan=0.0, posinf=0.0, neginf=0.0)

            return output.reshape(bsz, seq, d), aux_loss

        def break_symmetry(self, scale=1e-3):
            """Call only for random-init, NOT after loading checkpoint."""
            with torch.no_grad():
                self.router_bias.add_(torch.randn_like(self.router_bias) * scale)
                for exp in self.experts:
                    exp.break_symmetry(scale)

# ─────────────────────────────────────────────────────────────────────
# MTP Head
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class MTPHeadTorch(nn.Module):
        """FIX: lm_head weight TIED to embedding weight — no duplicate parameter."""

        def __init__(self, d_model):
            super().__init__()
            self.W1 = nn.Linear(d_model, d_model, bias=False)
            self.W2 = nn.Linear(d_model, d_model, bias=False)
            self.lm_head: Optional[nn.Parameter] = None

        def forward(self, x):
            h1 = F.silu(self.W1(x))
            h2 = F.silu(self.W2(h1))
            W  = self.lm_head
            if W is None:
                raise RuntimeError("MTPHeadTorch.lm_head not set; call tie_weights() first")
            l1 = h1 @ W.T
            l2 = h2 @ W.T
            return l1, l2

        def tie_weights(self, embedding_weight: "nn.Parameter"):
            self.lm_head = embedding_weight

# ─────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class TransformerBlockTorch(nn.Module):

        def __init__(self, layer_idx, d_model, n_heads, n_kv_heads, head_dim, d_ff, n_experts=8):
            super().__init__()
            self.layer_idx = layer_idx
            self.attn  = GQAttentionTorch(d_model, n_heads, n_kv_heads, head_dim)
            self.moe   = MoELayerTorch(d_model, d_ff, n_experts)
            self.norm1 = nn.RMSNorm(d_model)
            self.norm2 = nn.RMSNorm(d_model)

        def forward(self, x, cache,
                    add_noise: bool = False,
                    intent_bias: Optional[torch.Tensor] = None):
            h = self.attn(self.norm1(x), cache, self.layer_idx)
            x = x + h
            h, aux = self.moe(self.norm2(x),
                              add_noise=add_noise,
                              intent_bias=intent_bias)
            x = x + h
            return x, aux

# ─────────────────────────────────────────────────────────────────────
# Full PyTorch Model
# ─────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class DracoTransformerTorchV1(nn.Module):
        """
        PyTorch training model.
        After training, call _full_sync(numpy_model) to copy weights to
        the NumPy inference model.

        IMPORTANT — Training loop must use aux_total:
            l1, l2, aux_total = model(input_ids, return_aux=True)
            ce_loss  = F.cross_entropy(l1[:, :-1].reshape(-1, vocab), labels[:, 1:].reshape(-1))
            loss     = ce_loss + 0.01 * aux_total
            loss.backward()
        """

        def __init__(self, config: dict):
            super().__init__()
            self.config     = config
            self.d_model    = config.get("d_model",    128)
            self.n_layers   = config.get("n_layers",     4)
            self.n_heads    = config.get("n_heads",       4)
            self.n_kv_heads = config.get("n_kv_heads",   2)
            self.head_dim   = config.get("head_dim",    32)
            self.d_ff       = config.get("d_ff",       512)
            self.n_experts  = config.get("n_experts",    8)
            self.vocab_size = config.get("vocab_size", 151936)
            self.window     = config.get("window",    1024)

            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.lm_head   = self.embedding.weight  # tied
            self.blocks = nn.ModuleList([
                TransformerBlockTorch(
                    i, self.d_model, self.n_heads, self.n_kv_heads,
                    self.head_dim, self.d_ff, self.n_experts
                )
                for i in range(self.n_layers)
            ])
            self.norm_f = nn.RMSNorm(self.d_model)
            self.mtp = MTPHeadTorch(self.d_model)
            self.mtp.tie_weights(self.embedding.weight)

        def forward(self, token_ids: "torch.Tensor",
                    cache: Optional["KVCacheTorch"] = None,
                    return_aux: bool = True,
                    add_noise: bool = False,
                    intent_bias: Optional[torch.Tensor] = None):
            """
            token_ids: (batch, seq)
            Returns: (l1, l2, aux_total)
              l1:        (batch, seq, vocab)
              l2:        (batch, seq, vocab)
              aux_total: scalar tensor — sum of MoE load-balance losses

            FIX LOGITS-STABILITY: l1/l2 clamped to [-100, 100].
            intent_bias: optional (n_experts,) tensor — engine bias added to router logits.
            """
            x = self.embedding(token_ids)

            if cache is None:
                cache = KVCacheTorch(
                    self.n_layers, self.n_kv_heads, self.head_dim,
                    window=self.window, device=token_ids.device
                )

            aux_total = torch.tensor(0.0, device=x.device)
            for block in self.blocks:
                x, aux = block(x, cache,
                               add_noise=add_noise,
                               intent_bias=intent_bias)
                if return_aux:
                    aux_total = aux_total + aux

            x  = self.norm_f(x)
            l1 = x @ self.lm_head.T
            _, l2 = self.mtp(x)

            # FIX LOGITS-STABILITY
            l1 = torch.clamp(l1, -100.0, 100.0)
            l2 = torch.clamp(l2, -100.0, 100.0)

            cache.step(token_ids.shape[1])
            return l1, l2, aux_total

        def _full_sync(self, numpy_model) -> None:
            """
            Copy all weights from this PyTorch model to the NumPy inference model.
            FIX 3.1: Shape validation before every assignment.
            """
            def _t(tensor) -> np.ndarray:
                return tensor.detach().float().cpu().numpy()

            def _safe_assign(target_obj, attr: str, src_arr: np.ndarray, label: str):
                dst = getattr(target_obj, attr, None)
                if dst is None:
                    setattr(target_obj, attr, src_arr)
                    return
                if hasattr(dst, 'shape') and dst.shape != src_arr.shape:
                    raise ValueError(
                        f"_full_sync shape mismatch at '{label}': "
                        f"NumPy model expects {dst.shape}, "
                        f"PyTorch model provides {src_arr.shape}. "
                        f"Ensure both models use identical config dicts."
                    )
                setattr(target_obj, attr, src_arr)

            emb = _t(self.embedding.weight)
            _safe_assign(numpy_model, "embedding",   emb,  "embedding")
            numpy_model.lm_head       = numpy_model.embedding
            numpy_model.mtp.lm_head   = numpy_model.lm_head

            _safe_assign(numpy_model, "norm_f", _t(self.norm_f.weight), "norm_f")

            for i, (torch_blk, np_blk) in enumerate(zip(self.blocks, numpy_model.blocks)):
                pfx = f"block[{i}]"
                _safe_assign(np_blk, "norm1", _t(torch_blk.norm1.weight), f"{pfx}.norm1")
                _safe_assign(np_blk, "norm2", _t(torch_blk.norm2.weight), f"{pfx}.norm2")

                def _get_weight(module):
                    if isinstance(module, LoRALinear):
                        w = module.linear.weight
                        if not module.merged:
                            delta = (module.lora_B @ module.lora_A) * module.scaling
                            w = w + delta
                        return w
                    return module.weight

                # Attention: PyTorch Linear is (out, in); NumPy expects (in, out)
                _safe_assign(np_blk.attn, "W_q", _t(_get_weight(torch_blk.attn.W_q)).T, f"{pfx}.attn.W_q")
                _safe_assign(np_blk.attn, "W_k", _t(_get_weight(torch_blk.attn.W_k)).T, f"{pfx}.attn.W_k")
                _safe_assign(np_blk.attn, "W_v", _t(_get_weight(torch_blk.attn.W_v)).T, f"{pfx}.attn.W_v")
                _safe_assign(np_blk.attn, "W_o", _t(_get_weight(torch_blk.attn.W_o)).T, f"{pfx}.attn.W_o")

                for e, (torch_exp, np_exp) in enumerate(zip(torch_blk.moe.experts, np_blk.moe.experts)):
                    _safe_assign(np_exp, "W_g", _t(torch_exp.W_g.weight).T, f"{pfx}.expert[{e}].W_g")
                    _safe_assign(np_exp, "W_u", _t(torch_exp.W_u.weight).T, f"{pfx}.expert[{e}].W_u")
                    _safe_assign(np_exp, "W_d", _t(torch_exp.W_d.weight).T, f"{pfx}.expert[{e}].W_d")

                _safe_assign(np_blk.moe.shared, "W_g", _t(torch_blk.moe.shared.W_g.weight).T, f"{pfx}.shared.W_g")
                _safe_assign(np_blk.moe.shared, "W_u", _t(torch_blk.moe.shared.W_u.weight).T, f"{pfx}.shared.W_u")
                _safe_assign(np_blk.moe.shared, "W_d", _t(torch_blk.moe.shared.W_d.weight).T, f"{pfx}.shared.W_d")
                _safe_assign(np_blk.moe, "W_router",    _t(torch_blk.moe.W_router.weight).T, f"{pfx}.router.W")
                _safe_assign(np_blk.moe, "router_bias",  _t(torch_blk.moe.router_bias),       f"{pfx}.router.bias")
                _safe_assign(np_blk.moe, "norm_w", _t(torch_blk.moe.norm.weight), f"{pfx}.moe.norm_w")

            _safe_assign(numpy_model.mtp, "W1", _t(self.mtp.W1.weight).T, "mtp.W1")
            _safe_assign(numpy_model.mtp, "W2", _t(self.mtp.W2.weight).T, "mtp.W2")

        def enable_lora(self, rank: int = 8, alpha: float = 16.0,
                        target_modules: Optional[List[str]] = None):
            """
            FIX LORA: Replace attention projections with LoRALinear for PEFT.
            Freezes base model weights; only LoRA params are trainable.
            """
            if target_modules is None:
                target_modules = ["W_q", "W_v"]

            for p in self.parameters():
                p.requires_grad_(False)

            for blk in self.blocks:
                attn = blk.attn
                for attr in target_modules:
                    lin = getattr(attn, attr, None)
                    if lin is None or not isinstance(lin, nn.Linear):
                        continue
                    in_f  = lin.in_features
                    out_f = lin.out_features
                    lora  = LoRALinear(in_f, out_f, rank=rank, alpha=alpha, bias=False)
                    lora.linear.weight.data.copy_(lin.weight.data)
                    setattr(attn, attr, lora)

            for blk in self.blocks:
                attn = blk.attn
                for attr in target_modules:
                    mod = getattr(attn, attr, None)
                    if isinstance(mod, LoRALinear):
                        mod.lora_A.requires_grad_(True)
                        mod.lora_B.requires_grad_(True)

        def merge_lora(self):
            """Merge all LoRA adapters into base weights before _full_sync()."""
            for blk in self.blocks:
                for attr in ["W_q", "W_k", "W_v", "W_o"]:
                    mod = getattr(blk.attn, attr, None)
                    if isinstance(mod, LoRALinear):
                        mod.merge()

        def freeze_experts(self):
            """Freeze all expert FFN weights — only router is trainable.
            Use for router-only fine-tuning (improves load balance without
            modifying learned expert representations).
            Re-enable full training with unfreeze_experts()."""
            for blk in self.blocks:
                for exp in blk.moe.experts:
                    for p in exp.parameters():
                        p.requires_grad_(False)
                for p in blk.moe.shared.parameters():
                    p.requires_grad_(False)

        def unfreeze_experts(self):
            """Unfreeze all expert FFN weights (restore full training)."""
            for blk in self.blocks:
                for exp in blk.moe.experts:
                    for p in exp.parameters():
                        p.requires_grad_(True)
                for p in blk.moe.shared.parameters():
                    p.requires_grad_(True)

        def enable_qlora(self, rank: int = 8, alpha: float = 16.0,
                         target_modules: Optional[List[str]] = None,
                         compute_dtype: "torch.dtype" = None):
            """
            QLoRA: 4-bit quantised base weights + trainable LoRA adapters.
            Requires: pip install bitsandbytes

            How it works:
              1. Base linear weights are quantised to NF4 (4-bit) via bitsandbytes.
              2. LoRA low-rank adapters (rank × in + out × rank float16 params) are
                 added on top — only these are trained.
              3. Effective memory: ~0.5 bytes/param for base + small LoRA overhead.
              4. After training, call merge_qlora() to absorb LoRA into dequantised
                 weights, then _full_sync() to copy to NumPy inference model.

            target_modules: list of attn projection names to apply QLoRA on.
                            Default: ["W_q", "W_v"]  (standard PEFT practice)
            compute_dtype:  dtype for LoRA computation. Default: torch.float16.
                            Use torch.bfloat16 on Ampere+ GPUs.

            Usage:
                model = DracoTransformerTorchV1(config).cuda()
                model.enable_qlora(rank=16, alpha=32)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
                # Train normally with train_step(model, optimizer, ...)
                model.merge_qlora()
                model._full_sync(numpy_model)
            """
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "bitsandbytes not installed. Install with:\n"
                    "  pip install bitsandbytes\n"
                    "bitsandbytes requires CUDA — CPU-only machines cannot use QLoRA.\n"
                    "Alternative for CPU: use enable_lora() with standard float16."
                )

            if compute_dtype is None:
                compute_dtype = torch.float16
            if target_modules is None:
                target_modules = ["W_q", "W_v"]

            # Freeze everything first
            for p in self.parameters():
                p.requires_grad_(False)

            for blk in self.blocks:
                attn = blk.attn
                for attr in target_modules:
                    lin = getattr(attn, attr, None)
                    if lin is None:
                        continue
                    # Unwrap LoRALinear if already applied
                    if isinstance(lin, LoRALinear):
                        lin = lin.linear
                    if not isinstance(lin, nn.Linear):
                        continue
                    in_f  = lin.in_features
                    out_f = lin.out_features
                    # Create 4-bit quantised base layers
                    q_lin = bnb.nn.Linear4bit(
                        in_f, out_f,
                        bias=False,
                        compute_dtype=compute_dtype,
                        compress_statistics=True,
                        quant_type="nf4",
                    )
                    # Copy existing weights; bnb will quantise them on first forward()
                    q_lin.weight = bnb.nn.Params4bit(
                        lin.weight.data.to(compute_dtype),
                        requires_grad=False,
                        quant_type="nf4",
                    )
                    # LoRA adapters on top (float16, trainable)
                    lora = LoRALinear(in_f, out_f, rank=rank, alpha=alpha, bias=False)
                    lora.linear = q_lin          # swap base to 4-bit
                    lora.lora_A.data = lora.lora_A.data.to(compute_dtype)
                    lora.lora_B.data = lora.lora_B.data.to(compute_dtype)
                    lora.lora_A.requires_grad_(True)
                    lora.lora_B.requires_grad_(True)
                    setattr(attn, attr, lora)

        def merge_qlora(self):
            """Merge QLoRA adapters — dequantises 4-bit base then adds LoRA delta.
            After this call, the layers is a standard float16 nn.Linear.
            Required before _full_sync() so NumPy model gets full-precision weights.
            """
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("bitsandbytes not installed.")

            for blk in self.blocks:
                for attr in ["W_q", "W_k", "W_v", "W_o"]:
                    mod = getattr(blk.attn, attr, None)
                    if not isinstance(mod, LoRALinear):
                        continue
                    if not isinstance(mod.linear, bnb.nn.Linear4bit):
                        # Standard LoRA — use existing merge()
                        mod.merge()
                        continue
                    # Dequantise 4-bit base weight
                    with torch.no_grad():
                        w_fp = bnb.functional.dequantize_4bit(
                            mod.linear.weight.data,
                            mod.linear.weight.quant_state,
                        ).to(torch.float32)
                        if not mod.merged:
                            delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                            w_fp  = w_fp + delta.to(torch.float32)
                    # Replace with standard float16 nn.Linear
                    new_lin = nn.Linear(mod.linear.in_features,
                                        mod.linear.out_features, bias=False)
                    new_lin.weight.data.copy_(w_fp.to(torch.float16))
                    # Replace LoRALinear with plain merged linear
                    setattr(blk.attn, attr, new_lin)

        def export_gguf(self, output_path: str,
                        numpy_model_config: Optional[dict] = None) -> str:
            """
            Export this PyTorch model to GGUF FP16 via the NumPy GGUFExporter.
            Steps:
              1. Creates a temporary DracoTransformerV1 (NumPy) from this model's config.
              2. Calls _full_sync() to copy weights to it.
              3. Calls GGUFExporter.write_gguf() on the NumPy model.

            After export, quantise with llama.cpp:
                llama-quantize <output_path> out_q4km.gguf Q4_K_M

            Requires: transformer_v1.py importable (same directory or PYTHONPATH).
            """
            try:
                from transformer_v1 import DracoTransformerV1, GGUFExporter
            except ImportError:
                raise ImportError(
                    "transformer_v1.py must be importable to use export_gguf().\n"
                    "Ensure both files are in the same directory."
                )
            cfg = numpy_model_config or self.config
            np_model = DracoTransformerV1(cfg)
            self._full_sync(np_model)
            GGUFExporter(np_model).write_gguf(output_path)
            return output_path

        def load_external_weights(self, state_dict: dict, from_checkpoint: bool = True):
            """FIX: Expert AVERAGING (same logic as NumPy version)."""
            import re as _re

            expert_accum: Dict[int, Dict[str, List]] = {e: {} for e in range(self.n_experts)}
            shared_accum: Dict[str, List] = {}

            def _accum(d, k, arr):
                if k not in d:
                    d[k] = [arr.clone().float(), 1]
                else:
                    d[k][0] += arr.float()
                    d[k][1] += 1

            def _to_tensor(v):
                if isinstance(v, torch.Tensor):
                    return v
                return torch.tensor(np.array(v), dtype=torch.float32)

            for key, val in state_dict.items():
                t = _to_tensor(val)

                if "embed_tokens" in key:
                    self.embedding.weight.data.copy_(t)
                    continue
                if "lm_head" in key:
                    if t.shape == self.embedding.weight.shape:
                        self.embedding.weight.data.copy_(t)
                    continue
                if "model.norm.weight" in key:
                    self.norm_f.weight.data.copy_(t[:self.d_model])
                    continue

                for i, blk in enumerate(self.blocks):
                    tag = f"layers.{i}."
                    if tag not in key:
                        continue

                    def _copy_to(module, tensor):
                        target = module.linear if isinstance(module, LoRALinear) else module
                        sz = target.weight.shape
                        target.weight.data.copy_(tensor[:sz[0], :sz[1]])

                    if "q_proj" in key: _copy_to(blk.attn.W_q, t)
                    if "k_proj" in key: _copy_to(blk.attn.W_k, t)
                    if "v_proj" in key: _copy_to(blk.attn.W_v, t)
                    if "o_proj" in key: _copy_to(blk.attn.W_o, t)
                    if "input_layernorm"          in key: blk.norm1.weight.data.copy_(t[:self.d_model])
                    if "post_attention_layernorm" in key: blk.norm2.weight.data.copy_(t[:self.d_model])

                m = _re.search(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)", key)
                if m:
                    src_layer = int(m.group(1))
                    proj      = m.group(2)
                    eidx      = src_layer % self.n_experts
                    _accum(expert_accum[eidx], proj, t)
                    _accum(shared_accum,       proj, t)

            proj_map = {"gate_proj": "W_g", "up_proj": "W_u", "down_proj": "W_d"}
            for eidx in range(self.n_experts):
                for proj, (total, count) in expert_accum[eidx].items():
                    avg  = total / count
                    attr = proj_map[proj]
                    for blk in self.blocks:
                        lin = getattr(blk.moe.experts[eidx], attr)
                        sz  = lin.weight.shape
                        lin.weight.data.copy_(avg[:sz[0], :sz[1]])

            for proj, (total, count) in shared_accum.items():
                avg  = total / count
                attr = proj_map[proj]
                for blk in self.blocks:
                    lin = getattr(blk.moe.shared, attr)
                    sz  = lin.weight.shape
                    lin.weight.data.copy_(avg[:sz[0], :sz[1]])

            if not from_checkpoint:
                for blk in self.blocks:
                    blk.moe.break_symmetry()

        def save_checkpoint(self, path: str, step: int = 0):
            os.makedirs(path, exist_ok=True)
            torch.save({
                "step":   step,
                "model":  self.state_dict(),
                "config": self.config,
            }, os.path.join(path, f"ckpt_{step:06d}.pt"))

        @classmethod
        def load_checkpoint(cls, path: str, filename: Optional[str] = None,
                            current_config: Optional[dict] = None
                            ) -> Tuple["DracoTransformerTorchV1", int]:
            """
            Load a checkpoint.
            FIX 3.3: Config compatibility check before load_state_dict.
            FIX CHECKPOINT-LATEST: Sorts by mtime for non-zero-padded step numbers.
            """
            if filename is None:
                ckpts = [f for f in os.listdir(path) if f.startswith("ckpt_") and f.endswith(".pt")]
                if not ckpts:
                    raise FileNotFoundError(f"No checkpoints in {path}")
                ckpts_with_mtime = [
                    (f, os.path.getmtime(os.path.join(path, f))) for f in ckpts
                ]
                ckpts_with_mtime.sort(key=lambda x: x[1], reverse=True)
                filename = ckpts_with_mtime[0][0]

            data     = torch.load(os.path.join(path, filename), map_location="cpu",
                                   weights_only=False)  # FIX: suppress PyTorch 2.4+ FutureWarning
            ckpt_cfg = data.get("config", {})

            # FIX 3.3: Validate critical shape-determining config fields
            CRITICAL_KEYS = ("vocab_size", "d_model", "n_layers", "n_heads",
                             "n_kv_heads", "head_dim", "d_ff", "n_experts")
            if current_config is not None:
                mismatches = []
                for k in CRITICAL_KEYS:
                    ckpt_val = ckpt_cfg.get(k)
                    curr_val = current_config.get(k)
                    if ckpt_val is not None and curr_val is not None and ckpt_val != curr_val:
                        mismatches.append(f"  {k}: checkpoint={ckpt_val}, current={curr_val}")
                if mismatches:
                    raise ValueError(
                        f"Config mismatch between checkpoint '{filename}' and current_config:\n"
                        + "\n".join(mismatches)
                        + "\nEither use the checkpoint's config or convert the weights."
                    )

            model = cls(ckpt_cfg)
            model.load_state_dict(data["model"])
            return model, data.get("step", 0)

    # ── Training step helper ──────────────────────────────────────────

    def train_step(
        model: "DracoTransformerTorchV1",
        optimizer: "torch.optim.Optimizer",
        input_ids: "torch.Tensor",
        labels: "torch.Tensor",
        aux_coeff: float = 0.01,
        max_grad_norm: float = 1.0,
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    ) -> Optional[float]:
        """
        Single training step with:
        FIX GRADIENT-CLIP: gradient clipping to max_grad_norm.
        FIX NAN-GUARD: skips step if loss is NaN.
        FIX TRAIN-STEP (🔴 CRITICAL): Fixed torch.no_grad().__class__() syntax error.
            The old code used torch.no_grad().__class__() as a context manager,
            which raises AttributeError because no_grad() is a context manager
            instance, not a class.
            Fix: use contextlib.nullcontext() for the non-AMP path.
        Returns loss value or None if step was skipped due to NaN.
        """
        use_amp = scaler is not None

        # FIX TRAIN-STEP: use nullcontext() instead of torch.no_grad().__class__()
        amp_ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()

        optimizer.zero_grad()

        # FIX DDP-VOCAB: support both plain model and DDP-wrapped model
        base_model = model.module if hasattr(model, "module") else model

        with amp_ctx:
            l1, l2, aux_total = model(input_ids, return_aux=True)
            ce_loss = F.cross_entropy(
                l1[:, :-1].reshape(-1, base_model.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            # FIX MTP-TRAIN-SIGNAL: add auxiliary cross-entropy on l2 (next+1 prediction)
            # so MTPHead.W1/W2 receive proper gradients. Without this, W1 is only
            # trained indirectly via W2 and the speculative decoding quality degrades.
            mtp_coeff = getattr(base_model, "_mtp_aux_coeff", 0.1)
            if mtp_coeff > 0 and l2.shape[1] > 1:
                mtp_loss = F.cross_entropy(
                    l2[:, :-1].reshape(-1, base_model.vocab_size),
                    labels[:, 2:].reshape(-1) if labels.shape[1] > 2 else labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                loss = ce_loss + aux_coeff * aux_total + mtp_coeff * mtp_loss
            else:
                loss = ce_loss + aux_coeff * aux_total

        # FIX NAN-GUARD: skip step if loss is NaN
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            return None

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        return float(loss.item())

# ─────────────────────────────────────────────────────────────────────
# Recommended Training Loop (reference)
# ─────────────────────────────────────────────────────────────────────
#
# import torch, torch.nn.functional as F
# from transformer_torch_v1 import DracoTransformerTorchV1, train_step
#
# model     = DracoTransformerTorchV1(config).cuda()
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
# scaler    = torch.cuda.amp.GradScaler()
#
# AUX_COEFF = 0.01
#
# for step, batch in enumerate(dataloader):
#     input_ids = batch["input_ids"].cuda()
#     labels    = batch["labels"].cuda()
#
#     loss = train_step(model, optimizer, input_ids, labels,
#                       aux_coeff=AUX_COEFF, scaler=scaler)
#
#     if loss is None:
#         print(f"Step {step}: NaN loss — skipped")
#     else:
#         if step % 100 == 0:
#             print(f"Step {step}: loss={loss:.4f}")
#
#     if step % 1000 == 0:
#         model.save_checkpoint("checkpoints/", step=step)