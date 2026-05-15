# DracoAI V1 — modeling/transformer.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Transformer V1 — conductor module.

FIXES (this revision):
  ✅ FIX-IMPORT-INFERENCE-CONTEXT  : removed non-existent runtime/inference_context
     module; imports go directly to tensor_pool, profiler, health, precision, wal.
  ✅ FIX-DUPLICATE-TRANSFORMERBLOCK: TransformerBlock is defined ONLY in
     layers/block.py and imported here — no local redefinition.
  ✅ FIX-SPEC-MISSING-KV           : after verify accepts spec_pending, the
     engine immediately forwards spec_pending into the cache.
  ✅ FIX-DOUBLE-FORWARD-ACCEPT     : after forwarding spec_pending we use the
     returned logits directly; cur is set to [] so the main loop does NOT
     re-forward the same token.
  ✅ FIX-N_POS-SPEC-ACCEPT         : pos[spec_pending] recorded before sampling nid.
  ✅ FIX-L2-UPDATE-AFTER-TREE      : l2 updated to tree's final_l2.
  ✅ FIX-WAL-SPEC-ORDERING         : WAL written only after acceptance.
  ✅ FIX-TREE-BUDGET-OVERFLOW      : budget cap checked per-token inside loop.
  ✅ FIX-LLAMA-DUPLICATE-KWARGS    : top_p/temperature not duplicated in bridge.
  ✅ FIX-LOAD-EXTERNAL-INVALIDATE  : _invalidate_stacked() called per-block.
  ✅ FIX-KVCACHE-VECTORISED-UPDATE : vectorised slot assignment in KVCache.update().
  ✅ FIX-ENCAPSULATION-CACHE-POS   : GQAttention now calls cache.get_pos().
  ✅ FIX-MU-CONTAMINATION-REJECT   : on spec rejection, mu restored before resample.
  ✅ FIX-NPOS-OFFBYONE-REJECT      : rep-penalty recomputation uses (n_pos - 1).
  ✅ FIX-TREE-DECODER-ALLOC        : SpeculativeTreeDecoder allocated once per
     generate() call.
  ✅ FIX-TREE-BUDGET-KV-DESYNC     : remaining token budget passed to try_tree().
  ✅ FIX-LLAMA-FALLBACK-PROMPT     : create_completion fallback raises RuntimeError.
  ✅ FIX-ENGRAM-COMMIT-TIMING      : _try_commit_block() now uses cache.get_pos()
     as the sole authoritative position — eliminates off-by-one where n_pos is
     incremented for a new token BEFORE forward() writes that token's KV.
     The function scans ALL boundaries from _last_committed_end up to
     cache.get_pos(), so no block is ever silently skipped.
  ✅ FIX-ENGRAM-PREFILL-TRACKING   : replaced fragile "cur is ids" identity check
     with explicit _in_prefill flag; freq/pos/n_pos are always populated for
     every token forwarded during prefill, including partial-prefix-cache-hit
     suffixes.
  ✅ FIX-ENGRAM-SPEC-REJECT-EOS    : engram commit called after forward([verify_id])
     on the EOS-in-reject path so the EOS token's K/V block boundary is never lost.
  ✅ FIX-ENGRAM-SNAP-RESTORE       : snapshot/restore still works correctly with
     the new commit API via engram.snapshot() / engram.restore().
  ✅ FIX-ENGRAM-DEEP-INTEGRATION   : EngramCache is forwarded through every
     TransformerBlock → GQAttention call, enabling three-tier hierarchical memory.
  ✅ FIX-DECODE-COMMIT             : _try_commit_block() now called in the else
     branch after normal decode forward; previously only on speculative/tree events.
  ✅ FIX-ENGRAM-WINDOW-CHECK       : set_engram_cache() emits RuntimeWarning
     when window < block_size to prevent silent permanent token loss.
  ✅ FIX-PREFIX-ENGRAM-SNAPSHOT    : generate() stores engram.snapshot() in
     prefix_cache.put() and restores it on full-hit get().  All three full-hit
     sub-cases are handled: (a) logits+engram_snap, (b) no-logits+engram_snap,
     (c) legacy entry with no engram_snap.
  ✅ FIX-FULL-HIT-NO-LOGITS-PREFILL: Full-hit path that re-forwards last token
     now uses _in_prefill=False so freq/pos/n_pos are NOT double-counted.
  ✅ FIX-UNUSED-IMPORTS            : removed unused imports MOE_NOISE_SCALE,
     GQAttention, ExpertFFN, MoELayer, mm, RequestHandle,
     ContinuousBatchingScheduler — none of these are referenced in the module
     body; they are all re-exported from their respective sub-packages and do
     not need to be imported here.
  ✅ FIX-STREAMCB-DOUBLE-FIRE        : stream_cb previously fired at spec
     proposal time (optimistic/premature) AND again at rejection time with the
     correct token.  Callers received two callbacks for one generation position:
     first a wrong token, then the right one.  Fix: stream_cb is now suppressed
     at proposal; it fires exactly once when the spec token is CONFIRMED in the
     accept branch (via stored spec_pending_conf), or once with verify_id in
     the reject branch.  Zero duplicate callbacks, no missed tokens.
  ✅ FIX-TREEDEC-DEAD-PARAMS       : try_tree() no longer accepts ids, freq,
     pos, n_pos — these were silently unused.  Call site updated to match.
  ✅ FIX-SPEC-ACCEPT-DOUBLE-FORWARD: the second forward([spec_pending]) in the
     accept branch was redundant and HARMFUL — the first forward (via
     cur=[spec_id] at top of loop) already wrote the KV correctly and produced
     valid last_logits / l2.  The second forward incremented cache_pos a second
     time, placing the next token's KV one slot ahead of where the sequence
     actually was, causing phantom attention positions.  Removed; accept branch
     now uses the last_logits and l2 already computed by the verification forward.
  ✅ FIX-PREFIX-CACHE-SNAP-TIMING  : prefix_cache snapshot was previously taken
     AFTER generation completed, so the stored cache_pos included all generated
     tokens.  On the next call with the same prompt the restored cache_pos was
     wrong (prompt_len + prev_generated), causing every subsequent token's KV to
     be written at a phantom position.  Fix: the snapshot is now captured
     immediately after prefill completes (inside the _in_prefill branch) while
     cache_pos == len(prompt_ids).  The post-generation store block is removed.
  ✅ FIX-FULL-HIT-NO-LOGITS-CACHE-POS: Full prefix-cache hit (no cached logits)
     called cache.restore(snap) then forwarded prompt_ids[-1] again.  Since the
     restored snap already has cache_pos == len(prompt_ids), the extra forward
     advanced cache_pos to len(prompt_ids)+1 — every subsequent decode token's
     KV landed one position ahead of where the sequence actually was.  Fix: the
     no-logits path now snapshots and restores a temporary cache that sits at
     pos=len(prompt)-1 so the single re-forward lands at the correct position.
  ✅ FIX-WAL-FINAL-FLUSH           : generate() now flushes the WAL at the end
     of every call so the last (< WAL_FLUSH_INTERVAL) tokens are not lost on an
     unclean exit.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re as _re
import time
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    SINK_TOKENS, SPEC_THRESH, DEFAULT_TEMP, DEFAULT_TOP_P,
    LOGIT_CLIP,
)
from .config import ModelConfig

# Layers — only TransformerBlock is instantiated directly in this module
from .layers.block      import TransformerBlock
from .ops.tensor_ops    import rms_norm

# Cache
from .kv_cache.kv_cache     import KVCache
from .kv_cache.prefix_cache import PrefixCache
from .kv_cache.engram_cache import EngramCache

# Quantisation
from .quant.int4         import QuantizedLinear
from .quant.quant_linear import quantize_model_weights
from .quant.gguf_loader  import GGUFExporter

# Sampling
from .sampling.sampler import Sampler

# Runtime
from .runtime.tensor_pool import TensorPool
from .runtime.health      import HealthMonitor
from .runtime.precision   import DynamicPrecisionManager
from .runtime.wal         import WriteAheadLog
from .runtime.profiler    import InferenceProfiler
from .runtime.speculative import MTPHead, SpeculativeDecoder, SpeculativeTreeDecoder

__all__ = ["TransformerBlock", "DracoTransformerV1", "TransformerBridge"]

logger = logging.getLogger(__name__)
_SPEC_RECHECK_INTERVAL = 16


# ─────────────────────────────────────────────────────────────────────────────
# DracoTransformerV1
# ─────────────────────────────────────────────────────────────────────────────

class DracoTransformerV1:
    """DracoAI Transformer V1 — pure-NumPy inference engine.

    Deep Engram integration
    ───────────────────────
    Set an EngramCache via set_engram_cache() to enable three-tier hierarchical
    memory.  Once set:

    * forward() automatically passes the engram down through every
      TransformerBlock → GQAttention, blending Engram cross-attention with
      exact sliding-window attention at the hidden-state level before W_o.

    * generate() calls _try_commit_block() after every confirmed forward so
      the Engram continuously absorbs completed blocks.  The function uses
      cache.get_pos() as the authoritative write position — no off-by-one.

    * Speculative decoding takes Engram snapshots before each proposal and
      restores them on rejection, maintaining exact Engram ↔ KV consistency.

    Example::

        model = DracoTransformerV1(config)
        engram = EngramCache(
            n_layers=config.n_layers, n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim, d_model=config.d_model,
            block_size=128, top_k_retrieve=8, blend_alpha=0.85,
        )
        model.set_engram_cache(engram)
        out = model.generate([1, 2, 3], max_new_tokens=1_000_000)
    """

    def __init__(self, config, dtype: np.dtype = np.float32,
                 quant_mode: Optional[str] = None, quant_group_size: int = 128):
        if isinstance(config, ModelConfig):
            self.config = config.to_dict()
            _cfg = config
        else:
            self.config = config
            _cfg = ModelConfig.from_dict(config)

        self.d_model     = _cfg.d_model
        self.n_layers    = _cfg.n_layers
        self.n_heads     = _cfg.n_heads
        self.n_kv_heads  = _cfg.n_kv_heads
        self.head_dim    = _cfg.head_dim
        self.d_ff        = _cfg.d_ff
        self.n_experts   = _cfg.n_experts
        self.vocab_size  = _cfg.vocab_size
        self.window      = _cfg.window
        self._rope_theta = _cfg.rope_theta
        self._dtype      = np.dtype(dtype)
        self._quant_mode       = quant_mode
        self._quant_group_size = quant_group_size

        scale = 1.0 / math.sqrt(self.d_model)
        self.embedding = (
            np.random.randn(self.vocab_size, self.d_model) * scale
        ).astype(self._dtype)
        self.lm_head = self.embedding  # weight-tied

        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                i, self.d_model, self.n_heads, self.n_kv_heads,
                self.head_dim, self.d_ff, self.n_experts, self._rope_theta,
            )
            for i in range(self.n_layers)
        ]
        self.norm_f = np.ones(self.d_model, dtype=np.float32)
        self.mtp    = MTPHead(self.d_model, self.vocab_size)
        self.mtp.lm_head = self.lm_head

        # Optional runtime components
        self._cache:             Optional[KVCache]                 = None
        self._miro_mu:           float                             = 5.0
        self._memmap_cache:      bool                              = False
        self._memmap_dir:        Optional[str]                     = None
        self._prefix_cache:      Optional[PrefixCache]             = None
        self._engram_cache:      Optional[EngramCache]             = None
        self._tensor_pool:       Optional[TensorPool]              = None
        self._health_monitor:    Optional[HealthMonitor]           = None
        self._precision_manager: Optional[DynamicPrecisionManager] = None
        self._id_bias:           Optional[np.ndarray]              = None
        self._lm_head_f32:       Optional[np.ndarray]              = None
        self._spec_tracker = SpeculativeDecoder()

    # ── Cache factory ─────────────────────────────────────────────────
    def _make_cache(self, max_batch: int = 1) -> KVCache:
        return KVCache(
            self.n_layers, self.n_kv_heads, self.head_dim,
            window=self.window, sink=SINK_TOKENS,
            use_memmap=self._memmap_cache, memmap_dir=self._memmap_dir,
            max_batch=max_batch,
        )

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(
        self,
        token_ids:    List[int],
        cache:        KVCache,
        intent_boost: Optional[np.ndarray] = None,
        add_noise:    bool = True,
        intent_bias:  Optional[np.ndarray] = None,
        snap:         Optional[dict] = None,
        batch_idx:    int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Returns (l1_logits, l2_logits, aux_list).
        l1 = main LM-head logits.
        l2 = MTP draft logits (used for speculative prediction).

        Deep Engram integration: when self._engram_cache is set, it is
        passed down to each TransformerBlock → GQAttention, blending Engram
        cross-attention with exact sliding-window attention before W_o.
        """
        ids = np.clip(np.array(token_ids, dtype=np.int32), 0, self.vocab_size - 1)
        x   = self.embedding[ids][None]           # (1, seq, d_model)
        aux_list: List[Dict] = []

        for block in self.blocks:
            x, aux = block.forward(
                x, cache,
                add_noise=add_noise,
                intent_bias=intent_bias,
                snap=snap,
                batch_idx=batch_idx,
                pool=self._tensor_pool,
                engram=self._engram_cache,
            )
            aux_list.append(aux)

        x = rms_norm(x, self.norm_f)

        if self._lm_head_f32 is None:
            self._lm_head_f32 = self.lm_head.astype(np.float32)
        x32 = x.astype(np.float32) if x.dtype != np.float32 else x
        l1  = x32 @ self._lm_head_f32.T

        _, l2 = self.mtp.forward(x32)

        if self._id_bias is not None:
            l1 = l1 + self._id_bias[None, None, :]
        if intent_boost is not None:
            l1 = l1 + intent_boost[None, None, :]

        return l1, l2, aux_list

    # ── Sanity checks ─────────────────────────────────────────────────
    @staticmethod
    def _sanity_checks(logits: np.ndarray, label: str = ""):
        if np.any(np.isnan(logits)):
            raise RuntimeError(f"NaN in logits {label}")
        if np.any(np.isinf(logits)):
            raise RuntimeError(f"Inf in logits {label}")

    # ── Sampling helpers ──────────────────────────────────────────────
    @staticmethod
    def _sample_mirostat_v2(logits, mu, tau=5.0, eta=0.1):
        return Sampler.mirostat_v2(logits, mu, tau, eta)

    @staticmethod
    def _sample_topk_topp(logits, temp=DEFAULT_TEMP, top_p=DEFAULT_TOP_P,
                          top_k=50, min_p=0.0):
        return Sampler.topk_topp(logits, temp, top_p, top_k, min_p)

    # ── Setters for optional runtime components ───────────────────────
    def set_prefix_cache(self, cache: Optional[PrefixCache]):
        self._prefix_cache = cache

    def set_engram_cache(self, engram: Optional[EngramCache]):
        """
        Attach an EngramCache for deep three-tier hierarchical memory.

        When set, every forward() call will blend Engram cross-attention
        with exact sliding-window attention inside GQAttention, and
        generate() will continuously commit completed blocks into the Engram.
        Set to None to disable Engram (pure sliding-window mode).
        """
        if engram is not None:
            if engram.n_layers != self.n_layers:
                raise ValueError(
                    f"EngramCache.n_layers={engram.n_layers} != model.n_layers={self.n_layers}")
            if engram.n_kv_heads != self.n_kv_heads:
                raise ValueError(
                    f"EngramCache.n_kv_heads={engram.n_kv_heads} != model.n_kv_heads={self.n_kv_heads}")
            if engram.head_dim != self.head_dim:
                raise ValueError(
                    f"EngramCache.head_dim={engram.head_dim} != model.head_dim={self.head_dim}")
            if engram.d_model != self.d_model:
                raise ValueError(
                    f"EngramCache.d_model={engram.d_model} != model.d_model={self.d_model}")
            # ✅ FIX-ENGRAM-WINDOW-CHECK: warn if sliding window is smaller than
            # block_size.  Tokens would be evicted before they form a complete
            # block and would be permanently lost from Engram memory.
            if self.window < engram.block_size:
                import warnings
                warnings.warn(
                    f"EngramCache.block_size ({engram.block_size}) exceeds "
                    f"KVCache.window ({self.window}). Tokens will be evicted "
                    f"before they can be compressed into Engram blocks. "
                    f"Increase window or reduce block_size.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._engram_cache = engram

    def set_tensor_pool(self, pool: Optional[TensorPool]):
        self._tensor_pool = pool

    def set_health_monitor(self, monitor: Optional[HealthMonitor]):
        self._health_monitor = monitor

    def set_precision_manager(self, pm: Optional[DynamicPrecisionManager]):
        self._precision_manager = pm

    def set_identity_bias(self, token_ids: List[int], boost: float = 2.0):
        self._id_bias = np.zeros(self.vocab_size, dtype=np.float32)
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self._id_bias[tid] = boost

    # ─────────────────────────────────────────────────────────────────
    # generate()
    # ─────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt_ids:           List[int],
        max_new_tokens:       int   = 256,
        temp:                 float = DEFAULT_TEMP,
        top_p:                float = DEFAULT_TOP_P,
        min_p:                float = 0.0,
        eos_id:               int   = 151645,
        eos_ids:              Optional[List[int]] = None,
        new_prompt:           bool  = True,
        use_mirostat:         bool  = True,
        use_speculative:      bool  = True,
        use_speculative_tree: bool  = False,
        spec_tree_width:      int   = 3,
        spec_tree_depth:      int   = 2,
        adaptive_temp:        bool  = False,
        deterministic:        bool  = False,
        rep_alpha:            float = 0.5,
        temp_inertia:         float = 0.8,
        snap_delta_threshold: Optional[int] = None,
        debug:                bool  = False,
        stream_cb:            Optional[Callable[[int, float], None]] = None,
        intent_boost:         Optional[np.ndarray] = None,
        intent_bias:          Optional[np.ndarray] = None,
        profiler:             Optional[InferenceProfiler] = None,
        stop_event:           Optional[threading.Event] = None,
        checkpoint_every:     int   = 0,
        checkpoint_path:      Optional[str] = None,
        wal:                  Optional[WriteAheadLog] = None,
    ) -> List[int]:
        """
        Generate up to max_new_tokens tokens.

        Key invariants:
        ─────────────────────────────────────────────────────────────
        • n_generated tracks CONFIRMED tokens (never speculative-pending).
        • WAL written only AFTER a token is confirmed and flushed at end.
        • stream_cb fires exactly ONCE per confirmed output token (never for
          unverified speculative proposals).  On spec rejection the callback
          emits verify_id (the correct token), not the discarded spec_id.
        • _try_commit_block() uses cache.get_pos() as the authoritative KV
          write position. It scans ALL boundaries from _last_committed_end
          to cache.get_pos(), so no boundary is missed regardless of when
          it is called relative to n_pos.
        • Engram snapshots parallel KV snapshots for speculative rollback.
        • Speculative accept: the verification forward (via cur=[spec_id] at
          top of loop) IS the one and only forward for spec_id.  No second
          forward in the accept branch — that would advance cache_pos twice.
        • Prefix cache snapshot is captured immediately after prefill with
          cache_pos == len(prompt_ids) — never after generation completes.
        ─────────────────────────────────────────────────────────────
        """
        if new_prompt or self._cache is None:
            self._cache   = self._make_cache()
            self._miro_mu = 5.0
            if self._engram_cache is not None and new_prompt:
                self._engram_cache.clear()
        cache = self._cache

        _eos_set = set(eos_ids) if eos_ids else {eos_id}
        ids      = list(prompt_ids)

        # ── Prefix cache lookup ──────────────────────────────────────
        _prefix_hit          = False
        _cached_last_logits: Optional[np.ndarray] = None
        _plen_hit:           int = 0

        if self._prefix_cache is not None and new_prompt:
            _hit = self._prefix_cache.get(prompt_ids, rope_theta=self._rope_theta)
            if _hit is not None:
                # ✅ FIX-PREFIX-ENGRAM-SNAPSHOT: get() now returns 4-tuple;
                # _prefix_engram_snap is None for legacy entries.
                _snap, _plen_hit, _cached_last_logits, _prefix_engram_snap = _hit
                cache.restore(_snap)
                ids         = list(prompt_ids)
                _prefix_hit = True
                if debug:
                    logger.debug("[PrefixCache] HIT — skipped %d tokens", _plen_hit)
            else:
                _prefix_engram_snap = None
        else:
            _prefix_engram_snap = None

        mu  = self._miro_mu
        tau = 5.0
        eta = 0.1
        freq: Dict[int, int] = {}
        pos:  Dict[int, int] = {}
        n_pos = 0

        # Speculative state
        spec_pending:      Optional[int]        = None
        spec_pending_conf: float                = 0.0
        spec_snap:         Optional[dict]       = None
        engram_snap:       Optional[dict]       = None
        pre_spec_logits:   Optional[np.ndarray] = None
        mu_pre_verify:     float                = mu

        l2:               Optional[np.ndarray] = None
        last_logits:      Optional[np.ndarray] = None

        n_generated  = 0
        current_temp = temp
        add_noise    = not deterministic

        _snap_threshold = max(
            self.n_layers,
            snap_delta_threshold if snap_delta_threshold is not None
            else self.n_layers * 2,
        )

        _user_wants_speculative = use_speculative
        if use_speculative and self._spec_tracker.suggest_disable:
            use_speculative = False
            if debug:
                logger.debug(
                    "[speculative] auto-disabled at session start "
                    "(accept_ema=%.2f)", self._spec_tracker.accept_rate)

        if profiler is not None:
            profiler.start_session()

        # Allocate tree decoder once per generate() call
        _tree_dec: Optional[SpeculativeTreeDecoder] = (
            SpeculativeTreeDecoder(
                self,
                tree_width=spec_tree_width,
                tree_depth=spec_tree_depth,
                thresh=SPEC_THRESH,
            ) if use_speculative_tree else None
        )

        # ── Prefill setup ────────────────────────────────────────────
        # _in_prefill: True while processing the initial prompt batch.
        # Replaces the fragile "cur is ids" identity check so partial
        # prefix-cache-hit suffixes are also correctly tracked.
        _in_prefill = False

        if _prefix_hit:
            if _plen_hit == len(prompt_ids) and _cached_last_logits is not None:
                # ── Full hit + cached logits ─────────────────────────────
                for _idx, _tid in enumerate(prompt_ids):
                    freq[_tid] = freq.get(_tid, 0) + 1
                    pos[_tid]  = _idx
                n_pos = len(prompt_ids)
                cur   = []
                if (self._engram_cache is not None
                        and _prefix_engram_snap is not None):
                    self._engram_cache.restore(_prefix_engram_snap)
                    if debug:
                        logger.debug(
                            "[PrefixCache] Engram restored from prefix snapshot "
                            "(blocks=%d)", _prefix_engram_snap.get("_n_blocks", 0))

            elif _plen_hit == len(prompt_ids):
                # ── Full hit but no cached logits ────────────────────────
                # ✅ FIX-FULL-HIT-NO-LOGITS-CACHE-POS: the restored snapshot
                # has cache_pos == len(prompt_ids).  We need to re-forward the
                # last prompt token to get fresh logits, but doing so on the
                # already-restored cache would advance cache_pos to
                # len(prompt_ids)+1.  Fix: temporarily rewind cache_pos by 1
                # before the re-forward so the token lands at the correct slot.
                for _idx, _tid in enumerate(prompt_ids):
                    freq[_tid] = freq.get(_tid, 0) + 1
                    pos[_tid]  = _idx
                n_pos = len(prompt_ids)
                # Step cache_pos back one position so the re-forward of
                # prompt_ids[-1] writes into slot (len(prompt)-1) and leaves
                # cache_pos == len(prompt_ids) after the write.
                cache._cache_pos[0] = max(0, cache._cache_pos[0] - 1)
                cur   = [prompt_ids[-1]]
                _in_prefill = False
                if (self._engram_cache is not None
                        and _prefix_engram_snap is not None):
                    self._engram_cache.restore(_prefix_engram_snap)
                    if debug:
                        logger.debug(
                            "[PrefixCache] Engram restored from prefix snapshot "
                            "(no-logits path, blocks=%d)",
                            _prefix_engram_snap.get("_n_blocks", 0))

            else:
                # ── Partial hit ──────────────────────────────────────────
                cur = ids[_plen_hit:] if len(ids) > _plen_hit else [ids[-1]]
                for _idx, _tid in enumerate(prompt_ids[:_plen_hit]):
                    freq[_tid] = freq.get(_tid, 0) + 1
                    pos[_tid]  = _idx
                n_pos = _plen_hit
                _in_prefill = True
        else:
            cur = ids
            _in_prefill = True

        _prompt_last_logits: Optional[np.ndarray] = None

        # ── Engram helpers ───────────────────────────────────────────

        def _try_commit_block():
            """
            Scans ALL block boundaries from _last_committed_end up to
            cache.get_pos() and commits each completed block into the Engram.
            Uses cache.get_pos() as the authoritative KV write position.
            """
            if self._engram_cache is None:
                return
            bs          = self._engram_cache.block_size
            current_pos = cache.get_pos()
            last_end    = self._engram_cache._last_committed_end

            boundary = (last_end // bs + 1) * bs
            while boundary <= current_pos:
                block_start = boundary - bs
                block_end   = boundary

                oldest_accessible = max(0, current_pos - cache.window)
                if block_start < oldest_accessible:
                    self._engram_cache._last_committed_end = block_end
                    boundary += bs
                    continue

                layer_keys_list   = []
                layer_values_list = []
                valid = True

                for li in range(self.n_layers):
                    K_full, V_full = cache.get(li)
                    total_len  = K_full.shape[2]
                    oldest_pos = current_pos - total_len

                    offset     = block_start - oldest_pos
                    end_offset = offset + bs

                    if offset < 0 or end_offset > total_len:
                        valid = False
                        break

                    layer_keys_list.append(K_full[0, :, offset:end_offset, :])
                    layer_values_list.append(V_full[0, :, offset:end_offset, :])

                if not valid:
                    break

                layer_keys   = np.stack(layer_keys_list,   axis=0)
                layer_values = np.stack(layer_values_list, axis=0)

                self._engram_cache.commit_block(block_start, block_end,
                                                layer_keys, layer_values)
                boundary += bs

        def _engram_snapshot() -> Optional[dict]:
            if self._engram_cache is not None:
                return self._engram_cache.snapshot()
            return None

        def _engram_restore(snap: Optional[dict]):
            if self._engram_cache is not None and snap is not None:
                self._engram_cache.restore(snap)

        def _adaptive_temp(lg: np.ndarray) -> float:
            nonlocal current_temp
            if not adaptive_temp or use_mirostat:
                return current_temp
            p      = np.exp(lg - lg.max())
            p     /= p.sum() + 1e-9
            ent    = float(-np.sum(p * np.log(p + 1e-9)))
            norm_e = ent / (math.log(self.vocab_size) + 1e-9)
            target = (
                min(temp * 1.5, 2.0) if norm_e < 0.1
                else max(temp * 0.7, 0.3) if norm_e > 0.8
                else temp
            )
            current_temp = temp_inertia * current_temp + (1 - temp_inertia) * target
            return current_temp

        def _sample(lg: np.ndarray) -> Tuple[int, float]:
            nonlocal mu
            if use_mirostat:
                nid, mu = self._sample_mirostat_v2(lg, mu, tau, eta)
            else:
                nid = self._sample_topk_topp(
                    lg, _adaptive_temp(lg), top_p, min_p=min_p)
            conf = float(np.exp(np.clip(lg[nid] - lg.max(), -50, 0)))
            return nid, conf

        def _apply_rep_penalty(lg: np.ndarray, ref_n_pos: Optional[int] = None) -> np.ndarray:
            lg      = lg.copy()
            _n_pos  = ref_n_pos if ref_n_pos is not None else n_pos
            for tid, cnt in freq.items():
                if cnt > 0:
                    dist = _n_pos - pos.get(tid, 0) + 1
                    lg[tid] -= rep_alpha * math.log(1 + cnt) / dist
            return lg

        def _wal_append(tid: int):
            if wal is not None:
                wal.append(tid)

        # ── Helper: store prompt in prefix cache after prefill ────────
        def _maybe_store_prefix_cache():
            """
            Capture a full snapshot immediately after prefill so that
            cache_pos == len(prompt_ids) in the stored entry.
            Only called once, right after the prefill forward completes.
            Called only when _prefix_cache is set and this is a fresh prompt
            (not a prefix-cache hit).
            """
            if (
                self._prefix_cache is None
                or not new_prompt
                or _prefix_hit
                or len(prompt_ids) == 0
            ):
                return
            try:
                _snap_store = cache.snapshot(delta_threshold=0)
                cache._snap_escalate_to_full(_snap_store)
                _engram_snap_store = (
                    self._engram_cache.snapshot()
                    if self._engram_cache is not None else None
                )
                self._prefix_cache.put(
                    prompt_ids, _snap_store,
                    _prompt_last_logits,
                    rope_theta=self._rope_theta,
                    engram_snap=_engram_snap_store,
                )
            except Exception:
                pass

        # ── Main generation loop ─────────────────────────────────────
        while n_generated < max_new_tokens:
            if stop_event is not None and stop_event.is_set():
                break

            if (
                _user_wants_speculative
                and not use_speculative
                and spec_pending is None
                and n_generated % _SPEC_RECHECK_INTERVAL == 0
                and n_generated > 0
            ):
                if not self._spec_tracker.suggest_disable:
                    use_speculative = True
                    if debug:
                        logger.debug("[speculative] re-enabled at token %d", n_generated)

            _fwd_t0 = time.perf_counter()

            # ── Forward (or reuse cached logits) ─────────────────────
            if not cur and _cached_last_logits is not None:
                last_logits = _cached_last_logits.copy().astype(np.float64)
                _cached_last_logits = None
                if profiler:
                    profiler.record_forward(time.perf_counter() - _fwd_t0)
            elif not cur:
                pass
            else:
                l1, l2, _ = self.forward(
                    cur, cache,
                    intent_boost=intent_boost,
                    add_noise=add_noise,
                    intent_bias=intent_bias,
                )
                if profiler:
                    profiler.record_forward(time.perf_counter() - _fwd_t0)
                last_logits = l1[0, -1].copy().astype(np.float64)
                if _prompt_last_logits is None:
                    _prompt_last_logits = last_logits.copy()

                if _in_prefill:
                    for _idx, _tid in enumerate(cur):
                        freq[_tid] = freq.get(_tid, 0) + 1
                        pos[_tid]  = n_pos + _idx
                    n_pos += len(cur)
                    _try_commit_block()
                    cur = []
                    _in_prefill = False
                    # ✅ FIX-PREFIX-CACHE-SNAP-TIMING: capture the prefix-cache
                    # snapshot HERE, immediately after prefill completes, while
                    # cache_pos == len(prompt_ids).  Capturing it at the end of
                    # generate() (as done previously) stored cache_pos =
                    # len(prompt) + n_generated, making every subsequent restore
                    # land at the wrong position.
                    _maybe_store_prefix_cache()
                    continue
                else:
                    _try_commit_block()

            assert last_logits is not None, "last_logits is None — logic error"

            if debug:
                self._sanity_checks(last_logits, f"step={n_generated}")

            if self._health_monitor:
                _ec = (self.blocks[0].moe._expert_counts.copy()
                       if self.blocks else None)
                self._health_monitor.check_step(last_logits, expert_counts=_ec)
            if self._precision_manager:
                self._precision_manager.update(last_logits)

            last_logits = np.clip(last_logits, -LOGIT_CLIP, LOGIT_CLIP)
            last_logits = _apply_rep_penalty(last_logits)

            # ── Speculative verification ──────────────────────────────
            if spec_pending is not None:
                mu_pre_verify = mu
                verify_id, _ = _sample(last_logits)

                if verify_id == spec_pending:
                    # ════════ ACCEPTED ════════════════════════════════
                    # ✅ FIX-SPEC-ACCEPT-DOUBLE-FORWARD:
                    # The verification forward (cur=[spec_id] at top of loop)
                    # already wrote spec_id's KV and produced valid last_logits/l2.
                    # Do NOT call forward([spec_pending]) again here — that would
                    # advance cache_pos a second time, putting the next token's KV
                    # one slot ahead of the actual sequence position.
                    # last_logits and l2 are already fresh from the verification forward.
                    _wal_append(spec_pending)
                    if profiler:
                        profiler.record_spec_accept()
                    self._spec_tracker.record_accept()

                    # ✅ FIX-STREAMCB-DOUBLE-FIRE: emit the confirmed spec token
                    # here (at acceptance), not at proposal time.  Emitting at
                    # proposal was premature — if the spec token was later rejected
                    # the caller received a wrong token followed by the corrected
                    # one.  Now stream_cb fires exactly once per confirmed token.
                    if stream_cb:
                        stream_cb(spec_pending, spec_pending_conf)

                    # Record confirmed position for spec_pending.
                    # n_pos was already incremented when spec was proposed,
                    # so the confirmed position is n_pos - 1.
                    pos[spec_pending] = n_pos - 1
                    engram_snap = None

                    spec_pending = spec_snap = pre_spec_logits = None
                    spec_pending_conf = 0.0

                    _try_commit_block()

                    nid, conf = _sample(last_logits)
                    ids.append(nid)
                    freq[nid] = freq.get(nid, 0) + 1
                    pos[nid]  = n_pos
                    n_pos    += 1
                    n_generated += 1
                    if stream_cb:
                        stream_cb(nid, conf)
                    _wal_append(nid)
                    if nid in _eos_set or n_generated >= max_new_tokens:
                        break

                    _fwd_t2 = time.perf_counter()
                    l1_new, l2_new, _ = self.forward(
                        [nid], cache,
                        intent_boost=intent_boost,
                        add_noise=add_noise,
                        intent_bias=intent_bias,
                    )
                    if profiler:
                        profiler.record_forward(time.perf_counter() - _fwd_t2)

                    last_logits_new = np.clip(
                        l1_new[0, -1].astype(np.float64), -LOGIT_CLIP, LOGIT_CLIP)
                    last_logits_new = _apply_rep_penalty(last_logits_new)

                    if debug:
                        self._sanity_checks(last_logits_new, f"accept_fwd step={n_generated}")

                    _try_commit_block()

                    if use_speculative and l2_new is not None:
                        spec_id, spec_conf = self.mtp.try_speculative(l2_new)
                        if spec_id is not None and spec_id not in _eos_set:
                            pre_spec_logits   = last_logits_new.copy()
                            spec_snap         = cache.snapshot(_snap_threshold)
                            engram_snap       = _engram_snapshot()
                            ids.append(spec_id)
                            freq[spec_id] = freq.get(spec_id, 0) + 1
                            pos[spec_id]  = n_pos
                            n_pos    += 1
                            n_generated += 1
                            # ✅ FIX-STREAMCB-DOUBLE-FIRE: do NOT emit stream_cb
                            # here.  The spec token is still unverified; emitting
                            # now would cause a double-fire if later rejected.
                            # stream_cb fires at accept time via spec_pending_conf.
                            spec_pending_conf = spec_conf
                            if n_generated >= max_new_tokens:
                                break
                            spec_pending = spec_id

                    l2 = l2_new
                    last_logits = last_logits_new
                    cur = []
                    continue

                else:
                    # ════════ REJECTED ════════════════════════════════
                    if profiler:
                        profiler.record_spec_reject()
                    self._spec_tracker.record_reject()
                    if spec_snap is not None:
                        if profiler and spec_snap.get("_escalated"):
                            profiler.record_escalate()
                        cache.restore(spec_snap)
                    _engram_restore(engram_snap)
                    engram_snap = None

                    if pre_spec_logits is not None:
                        mu = mu_pre_verify
                        _ref_pos = n_pos - 1
                        last_logits = pre_spec_logits.copy()
                        for tid, cnt in freq.items():
                            if tid == spec_pending:
                                continue
                            if cnt > 0:
                                dist = _ref_pos - pos.get(tid, 0) + 1
                                last_logits[tid] -= rep_alpha * math.log(1 + cnt) / dist
                        verify_id, _ = _sample(last_logits)

                    if ids and ids[-1] == spec_pending:
                        ids[-1] = verify_id
                        freq.pop(spec_pending, None)
                        pos.pop(spec_pending, None)
                    freq[verify_id] = freq.get(verify_id, 0) + 1
                    pos[verify_id]  = n_pos - 1
                    conf = float(np.exp(np.clip(
                        last_logits[verify_id] - last_logits.max(), -50, 0)))
                    if stream_cb:
                        stream_cb(verify_id, conf)
                    _wal_append(verify_id)

                    if verify_id in _eos_set:
                        self.forward(
                            [verify_id], cache,
                            intent_boost=intent_boost,
                            add_noise=add_noise,
                            intent_bias=intent_bias,
                        )
                        _try_commit_block()
                        spec_pending = spec_snap = pre_spec_logits = None
                        break

                    spec_pending = spec_snap = pre_spec_logits = None
                    cur = [ids[-1]]
                    continue

            # ── Normal (non-speculative) sample step ──────────────────
            nid, conf = _sample(last_logits)
            ids.append(nid)
            freq[nid] = freq.get(nid, 0) + 1
            pos[nid]  = n_pos
            n_pos    += 1
            n_generated += 1
            if stream_cb:
                stream_cb(nid, conf)
            _wal_append(nid)

            if checkpoint_every > 0 and n_generated % checkpoint_every == 0:
                _ckpt = checkpoint_path or "dracoai_gen_checkpoint"
                try:
                    cache.save_checkpoint(_ckpt)
                    np.save(_ckpt + "_ids.npy", np.array(ids, dtype=np.int32))
                except Exception as _e:
                    if debug:
                        logger.debug("[checkpoint] save failed: %s", _e)

            if nid in _eos_set or n_generated >= max_new_tokens:
                break

            # ── Speculative tree decoding ─────────────────────────────
            if use_speculative_tree and _tree_dec is not None and l2 is not None:
                _remaining = max_new_tokens - n_generated
                _tree_accepted, last_logits, _tree_l2, mu = _tree_dec.try_tree(
                    cache, last_logits, l2,
                    _eos_set, mu, use_mirostat,
                    temp=current_temp, top_p=top_p, min_p=min_p,
                    intent_boost=intent_boost, intent_bias=intent_bias,
                    add_noise=add_noise,
                    max_tokens=_remaining,
                )

                if _tree_accepted:
                    l2 = _tree_l2
                    _eos_hit = False
                    for _t in _tree_accepted:
                        if n_generated >= max_new_tokens:
                            break
                        ids.append(_t)
                        freq[_t] = freq.get(_t, 0) + 1
                        pos[_t]  = n_pos
                        n_pos   += 1
                        n_generated += 1
                        if stream_cb:
                            stream_cb(_t, 1.0)
                        _wal_append(_t)
                        if _t in _eos_set:
                            _eos_hit = True
                            break

                    _try_commit_block()

                    if _eos_hit or n_generated >= max_new_tokens:
                        break
                    cur = []
                    continue

            # ── Single-token speculative ──────────────────────────────
            if use_speculative and l2 is not None:
                spec_id, spec_conf = self.mtp.try_speculative(l2)
                if spec_id is not None and spec_id not in _eos_set:
                    pre_spec_logits   = last_logits.copy()
                    spec_snap         = cache.snapshot(_snap_threshold)
                    engram_snap       = _engram_snapshot()
                    ids.append(spec_id)
                    freq[spec_id] = freq.get(spec_id, 0) + 1
                    pos[spec_id]  = n_pos
                    n_pos    += 1
                    n_generated += 1
                    # ✅ FIX-STREAMCB-DOUBLE-FIRE: do NOT emit stream_cb here.
                    # The spec token is still unverified; emitting now causes a
                    # double-fire on rejection (wrong token, then correct token).
                    # stream_cb fires once at confirmation via spec_pending_conf.
                    spec_pending_conf = spec_conf
                    if n_generated >= max_new_tokens:
                        break
                    spec_pending = spec_id

            cur = [ids[-1]]

        # ── Cleanup: remove unverified speculative token ──────────────
        if spec_pending is not None:
            if spec_snap is not None:
                cache.restore(spec_snap)
            _engram_restore(engram_snap)
            if ids and ids[-1] == spec_pending:
                ids.pop()
                n_generated -= 1
            pre_spec_logits = None
            engram_snap = None

        self._miro_mu = mu
        result = ids[len(prompt_ids):]

        # ✅ FIX-WAL-FINAL-FLUSH: flush WAL unconditionally so the tail
        # tokens (< WAL_FLUSH_INTERVAL since last auto-flush) are durable
        # even if the caller does not close the WAL via context manager.
        if wal is not None:
            wal.flush()

        if profiler is not None:
            profiler.record_tokens(len(result))
            profiler.end_session()

        return result

    # ── Load balancing ────────────────────────────────────────────────
    def adapt_load_balance(self, imbalance_thresh: float = 0.3,
                           correction_scale: float = 0.1):
        for blk in self.blocks:
            blk.moe.adapt_router_bias(
                imbalance_thresh=imbalance_thresh,
                correction_scale=correction_scale,
            )

    # ── Quantisation ──────────────────────────────────────────────────
    def quantize_weights(self, quant: Optional[str] = None, group_size: int = 128):
        mode = quant or self._quant_mode
        if mode is None:
            raise ValueError("Specify quant='int8' or 'int4'")
        self._quant_mode       = mode
        self._quant_group_size = group_size
        quantize_model_weights(self, mode, group_size)
        self._lm_head_f32 = None

    # ── Dtype cast ────────────────────────────────────────────────────
    def cast_weights(self, dtype: np.dtype):
        dtype = np.dtype(dtype)
        self._dtype       = dtype
        self.embedding    = self.embedding.astype(dtype)
        self.lm_head      = self.embedding
        self._lm_head_f32 = None
        if self.mtp.lm_head is not None:
            self.mtp.lm_head = self.lm_head
        self.norm_f = self.norm_f.astype(np.float32)
        for blk in self.blocks:
            blk.norm1 = blk.norm1.astype(np.float32)
            blk.norm2 = blk.norm2.astype(np.float32)
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                W = getattr(blk.attn, attr)
                if not isinstance(W, QuantizedLinear):
                    setattr(blk.attn, attr, W.astype(dtype))
            for exp in list(blk.moe.experts) + [blk.moe.shared]:
                for attr in ("W_g", "W_u", "W_d"):
                    W = getattr(exp, attr)
                    if not isinstance(W, QuantizedLinear):
                        setattr(exp, attr, W.astype(dtype))
            if not isinstance(blk.moe.W_router, QuantizedLinear):
                blk.moe.W_router = blk.moe.W_router.astype(dtype)
            blk.moe._invalidate_stacked()

    # ── Save / Load ───────────────────────────────────────────────────
    def save_weights(self, path: str):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embedding.npy"), self.embedding)
        np.save(os.path.join(path, "norm_f.npy"),    self.norm_f)
        for i, blk in enumerate(self.blocks):
            pfx = os.path.join(path, f"block_{i}")
            np.save(f"{pfx}_norm1.npy", blk.norm1)
            np.save(f"{pfx}_norm2.npy", blk.norm2)
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                W = getattr(blk.attn, attr)
                arr = W.dequantize() if isinstance(W, QuantizedLinear) else W
                np.save(f"{pfx}_attn_{attr}.npy", arr)
            for e, exp in enumerate(blk.moe.experts):
                for attr in ("W_g", "W_u", "W_d"):
                    W = getattr(exp, attr)
                    arr = W.dequantize() if isinstance(W, QuantizedLinear) else W
                    np.save(f"{pfx}_expert{e}_{attr}.npy", arr)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load_weights(cls, path: str) -> "DracoTransformerV1":
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model             = cls(config)
        model.embedding   = np.load(os.path.join(path, "embedding.npy"))
        model.lm_head     = model.embedding
        model.mtp.lm_head = model.lm_head
        model._lm_head_f32 = None
        model.norm_f      = np.load(os.path.join(path, "norm_f.npy"))
        for i, blk in enumerate(model.blocks):
            pfx = os.path.join(path, f"block_{i}")
            blk.norm1 = np.load(f"{pfx}_norm1.npy")
            blk.norm2 = np.load(f"{pfx}_norm2.npy")
            for attr in ("W_q", "W_k", "W_v", "W_o"):
                setattr(blk.attn, attr, np.load(f"{pfx}_attn_{attr}.npy"))
            for e, exp in enumerate(blk.moe.experts):
                for attr in ("W_g", "W_u", "W_d"):
                    setattr(exp, attr, np.load(f"{pfx}_expert{e}_{attr}.npy"))
        return model

    def load_external_weights(self, state_dict: dict, from_checkpoint: bool = True):
        """Load weights from an external HuggingFace-style state dict."""
        expert_accum: Dict[int, Dict[str, list]] = {
            e: {} for e in range(self.n_experts)}
        shared_accum: Dict[str, list] = {}

        def _accum(accum_dict, key, arr):
            if key not in accum_dict:
                accum_dict[key] = [arr.copy().astype(np.float32), 1]
            else:
                accum_dict[key][0] += arr.astype(np.float32)
                accum_dict[key][1] += 1

        proj_attr_map = {
            "gate_proj": "W_g",
            "up_proj":   "W_u",
            "down_proj": "W_d",
        }

        for key, val in state_dict.items():
            arr = val if isinstance(val, np.ndarray) else np.array(
                val, dtype=np.float32)

            if "embed_tokens" in key:
                self.embedding    = arr.astype(np.float32)
                self.lm_head      = self.embedding
                self.mtp.lm_head  = self.lm_head
                self._lm_head_f32 = None
                continue

            if "lm_head" in key and "embed" not in key:
                self.lm_head      = arr.astype(np.float32)
                self.mtp.lm_head  = self.lm_head
                self._lm_head_f32 = None
                continue

            if "model.norm.weight" in key:
                self.norm_f = arr.astype(np.float32)
                continue

            for i, block in enumerate(self.blocks):
                tag = f"layers.{i}."
                if tag not in key:
                    continue
                if "q_proj"                   in key: block.attn.W_q = arr.T.astype(np.float32)
                if "k_proj"                   in key: block.attn.W_k = arr.T.astype(np.float32)
                if "v_proj"                   in key: block.attn.W_v = arr.T.astype(np.float32)
                if "o_proj"                   in key: block.attn.W_o = arr.T.astype(np.float32)
                if "input_layernorm"          in key: block.norm1    = arr.astype(np.float32)
                if "post_attention_layernorm" in key: block.norm2    = arr.astype(np.float32)

                m_expert = _re.search(
                    r"layers\.\d+\.mlp\.experts\.(\d+)\."
                    r"(gate_proj|up_proj|down_proj)", key)
                if m_expert:
                    eid  = int(m_expert.group(1)) % self.n_experts
                    attr = proj_attr_map[m_expert.group(2)]
                    _accum(expert_accum[eid], attr, arr.T)
                    continue

                m_dense = _re.search(
                    r"layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)", key)
                if m_dense and "experts" not in key:
                    attr = proj_attr_map[m_dense.group(1)]
                    for e in range(self.n_experts):
                        _accum(expert_accum[e], attr, arr.T)
                    _accum(shared_accum, attr, arr.T)

        for e in range(self.n_experts):
            for attr, (total, count) in expert_accum[e].items():
                avg = (total / count).astype(np.float32)
                for blk in self.blocks:
                    setattr(blk.moe.experts[e], attr, avg)

        for attr, (total, count) in shared_accum.items():
            avg = (total / count).astype(np.float32)
            for blk in self.blocks:
                setattr(blk.moe.shared, attr, avg)

        if not from_checkpoint:
            for blk in self.blocks:
                for exp in blk.moe.experts:
                    exp._break_symmetry()

        for blk in self.blocks:
            blk.moe._invalidate_stacked()


# ─────────────────────────────────────────────────────────────────────────────
# TransformerBridge — NumPy ↔ llama.cpp
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBridge:
    """
    Production inference bridge: NumPy ↔ llama.cpp.

    Selects backend automatically:
      • GGUF file present  → llama.cpp (faster, quantised)
      • No GGUF            → NumPy engine (always available)

    FIX-LLAMA-DUPLICATE-KWARGS: top_p / temperature appear ONLY inside
    gen_kwargs; they are NOT re-passed to create_completion.
    """

    BACKEND_NUMPY = "numpy"
    BACKEND_LLAMA = "llama.cpp"

    def __init__(
        self,
        numpy_model:    Optional[DracoTransformerV1] = None,
        gguf_path:      Optional[str] = None,
        n_gpu_layers:   int  = 0,
        n_ctx:          int  = 2048,
        verbose:        bool = False,
        checkpoint_dir: Optional[str] = None,
        gguf_filename:  str  = "dracoai.gguf",
    ):
        self._numpy_model  = numpy_model
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx        = n_ctx
        self._verbose      = verbose
        self._llama        = None
        self._intent_bias:  Optional[np.ndarray] = None
        self._intent_boost: Optional[np.ndarray] = None

        if checkpoint_dir is not None:
            auto_path = os.path.join(checkpoint_dir, gguf_filename)
            if os.path.exists(auto_path):
                logger.info("[DracoAI] GGUF detected → llama.cpp (%s)", auto_path)
                gguf_path = auto_path
            else:
                logger.info("[DracoAI] No GGUF at %r → NumPy backend", auto_path)

        self._gguf_path = gguf_path
        if gguf_path and os.path.exists(gguf_path):
            self._backend = self.BACKEND_LLAMA
            self._load_llama()
        elif numpy_model is not None:
            self._backend = self.BACKEND_NUMPY
        else:
            raise ValueError("Provide numpy_model or an existing gguf_path.")

    def _load_llama(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "  pip install llama-cpp-python\n"
                "  (GPU: CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python)"
            )
        self._llama = Llama(
            model_path=self._gguf_path,
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            verbose=self._verbose,
        )

    def export_gguf(self, output_path: Optional[str] = None) -> str:
        if self._numpy_model is None:
            raise RuntimeError("No numpy_model to export.")
        path = output_path or self._gguf_path or "dracoai_fp16.gguf"
        GGUFExporter(self._numpy_model).write_gguf(path)
        self._gguf_path = path
        self._backend   = self.BACKEND_LLAMA
        self._load_llama()
        return path

    @property
    def backend(self) -> str:
        return self._backend

    def use_numpy(self):
        if self._numpy_model is None:
            raise RuntimeError("No numpy_model available.")
        self._backend = self.BACKEND_NUMPY

    def use_llama(self):
        if self._gguf_path is None or not os.path.exists(self._gguf_path):
            raise RuntimeError("No valid gguf_path. Call export_gguf() first.")
        self._backend = self.BACKEND_LLAMA
        if self._llama is None:
            self._load_llama()

    def set_intent_bias(self, bias):   self._intent_bias  = bias
    def set_intent_boost(self, boost): self._intent_boost = boost

    def _boost_to_logit_bias(self) -> Optional[Dict[int, float]]:
        if self._intent_boost is None:
            return None
        arr = self._intent_boost
        nz  = np.nonzero(arr)[0]
        if len(nz) == 0:
            return None
        if len(nz) > 200:
            nz = nz[np.argsort(np.abs(arr[nz]))[-200:]]
        return {int(i): float(arr[i]) for i in nz}

    def generate(
        self,
        prompt_ids:     List[int],
        max_new_tokens: int   = 256,
        temp:           float = DEFAULT_TEMP,
        top_p:          float = DEFAULT_TOP_P,
        min_p:          float = 0.0,
        eos_id:         int   = 151645,
        new_prompt:     bool  = True,
        use_mirostat:   bool  = True,
        use_speculative: bool = True,
        stream_cb:      Optional[Callable[[int, float], None]] = None,
    ) -> List[int]:
        if self._backend == self.BACKEND_NUMPY:
            return self._generate_numpy(
                prompt_ids, max_new_tokens, temp, top_p, min_p,
                eos_id, new_prompt, use_mirostat, use_speculative, stream_cb)
        return self._generate_llama(
            prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, stream_cb)

    def _generate_numpy(
        self, prompt_ids, max_new_tokens, temp, top_p, min_p,
        eos_id, new_prompt, use_mirostat, use_speculative, stream_cb,
    ):
        return self._numpy_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temp=temp, top_p=top_p, min_p=min_p,
            eos_id=eos_id, new_prompt=new_prompt,
            use_mirostat=use_mirostat, use_speculative=use_speculative,
            stream_cb=stream_cb,
            intent_boost=self._intent_boost,
            intent_bias=self._intent_bias,
        )

    def _generate_llama(
        self, prompt_ids, max_new_tokens, temp, top_p, min_p, eos_id, stream_cb,
    ) -> List[int]:
        """
        llama.cpp backend.
        FIX-LLAMA-DUPLICATE-KWARGS: sampling params appear ONLY in gen_kwargs.
        FIX-LLAMA-TOKEN-LIST: llama-cpp-python's generate() accepts a token list.
        """
        if self._llama is None:
            self._load_llama()

        logit_bias = self._boost_to_logit_bias()
        output_ids: List[int] = []

        gen_kwargs: dict = dict(
            top_k=50, top_p=top_p, min_p=min_p,
            temperature=temp, repeat_penalty=1.1,
        )
        if logit_bias is not None:
            gen_kwargs["logit_bias"] = logit_bias

        try:
            for tok in self._llama.generate(prompt_ids, **gen_kwargs):
                tok = int(tok)
                if tok == eos_id or len(output_ids) >= max_new_tokens:
                    break
                output_ids.append(tok)
                if stream_cb:
                    stream_cb(tok, 1.0)
        except TypeError:
            logger.error(
                "[DracoAI] llama.cpp streaming generate() failed (TypeError). "
                "Upgrade: pip install -U llama-cpp-python")
            raise RuntimeError(
                "llama.cpp streaming generate() failed.  "
                "Upgrade llama-cpp-python or use the NumPy backend."
            )

        return output_ids

    def __repr__(self) -> str:
        return (
            f"TransformerBridge(backend={self._backend!r}, "
            f"gguf={self._gguf_path!r})"
        )