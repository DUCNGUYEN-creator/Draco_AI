# DracoAI V1 — modeling/kv_cache/engram_cache.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Engram Cache — three-tier hierarchical compressed memory for ultra-long contexts.

Architecture (three tiers):
──────────────────────────────────────────────────────────────────────────────
  Tier 0  │ Exact KV   │ kv_cache.py SWA-Sink ring (recent WINDOW tokens)
  Tier 1  │ Engram     │ Per-block compressed keys/values (block_size tokens
           │            │ → 1 summary vector). Compression ratio = block_size.
  Tier 2  │ ToC        │ Table-of-Contents: mean of n_toc_blocks engram vectors
           │            │ → 1 chapter vector. Enables O(1) coarse lookup.
──────────────────────────────────────────────────────────────────────────────

FIXES in this revision:
  ✅ FIX-ENGRAM-COMMIT              : commit_block() API; generate() passes
     pre-sliced K/V tensors directly while block is still in window.
  ✅ FIX-ENGRAM-TEMPORAL-OFFSET     : _extract_block_from_cache() uses simple
     temporal offset into time-ordered tensor.
  ✅ FIX-ENGRAM-DEADLOCK            : _rebuild_toc_nolock() called only while
     lock is already held; no nested locking anywhere.
  ✅ FIX-MATRIX-DIRTY-TOC           : _rebuild_toc_nolock() sets _matrix_dirty.
  ✅ VECTORIZED-SCORE-BLOCKS        : pure-NumPy BLAS + argpartition O(n).
  ✅ MAX-POOL-SUMMARY               : sign-preserving max-pool keys per layer.
  ✅ DYNAMIC-BLEND-ALPHA            : attend() returns (eng_out, eff_alpha).
  ✅ FIX-IMPORTANCE-NORMALIZE       : sigmoid-normalised importance weight.
  ✅ FIX-ATTEND-PREFILL             : attend() handles seq > 1 (prefill).
  ✅ FIX-LAST-COMMITTED-EXPOSE      : _last_committed_end is a regular attr.
  ✅ FIX-O1-EVICTION                : deque + popleft() O(1) eviction.
  ✅ FIX-THREAD-SAFETY-COMMITTED-END: _last_committed_end written atomically
     INSIDE _add_block() while the lock is already held.
  ✅ FIX-UNUSED-VAR-SEQ             : removed unused local variable `seq` in
     attend().  Q.shape[2] was stored but never referenced in the method body;
     the query mean is computed directly from Q[0] without needing seq.
"""
from __future__ import annotations

import collections
import logging
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .kv_cache import KVCache

__all__ = ["EngramCache", "EngramBlock"]

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EngramBlock:
    """
    One compressed block representing `block_size` contiguous tokens.
    """
    start_pos:         int
    end_pos:           int
    layer_keys_mean:   np.ndarray   # (n_layers, n_kv_heads, head_dim)
    layer_keys_max:    np.ndarray   # (n_layers, n_kv_heads, head_dim)
    layer_values_mean: np.ndarray   # (n_layers, n_kv_heads, head_dim)
    summary:           np.ndarray   # (head_dim,)
    importance:        float = 0.0


# ── EngramCache ───────────────────────────────────────────────────────────────

class EngramCache:
    """
    Hierarchical compressed KV memory with vectorized retrieval,
    max-pool summary, and dynamic blend alpha.
    """

    def __init__(
        self,
        n_layers:          int,
        n_kv_heads:        int,
        head_dim:          int,
        d_model:           int,
        block_size:        int   = 128,
        top_k_retrieve:    int   = 8,
        n_toc_blocks:      int   = 8,
        blend_alpha:       float = 0.85,
        dynamic_alpha:     bool  = True,
        alpha_high_thresh: float = 0.85,
        alpha_low_thresh:  float = 0.50,
        max_pool_weight:   float = 0.20,
        dtype:             np.dtype = np.float16,
        max_blocks:        Optional[int] = None,
    ):
        self.n_layers          = n_layers
        self.n_kv_heads        = n_kv_heads
        self.head_dim          = head_dim
        self.d_model           = d_model
        self.block_size        = block_size
        self.top_k_retrieve    = top_k_retrieve
        self.n_toc_blocks      = n_toc_blocks
        self.blend_alpha       = blend_alpha
        self.dynamic_alpha     = dynamic_alpha
        self.alpha_high_thresh = alpha_high_thresh
        self.alpha_low_thresh  = alpha_low_thresh
        self.max_pool_weight   = max_pool_weight
        self.dtype             = np.dtype(dtype)
        self.max_blocks        = max_blocks

        self._blocks: collections.deque = collections.deque()
        self._lock   = threading.Lock()

        self._summary_matrix: Optional[np.ndarray] = None
        self._importance_vec: Optional[np.ndarray] = None
        self._matrix_dirty   = True

        self._toc_summaries: Optional[np.ndarray] = None
        self._toc_block_idx: List[Tuple[int, int]] = []

        scale = 1.0 / math.sqrt(head_dim)
        self._W_summary = np.eye(head_dim, dtype=np.float32) * scale

        # ✅ FIX-THREAD-SAFETY-COMMITTED-END: _last_committed_end is updated
        # INSIDE _add_block() while the lock is held, so block visibility and
        # pointer advancement are always atomic w.r.t. the lock.
        self._last_committed_end: int = 0
        self._n_commits   = 0
        self._n_retrieves = 0

    # ── Primary commit API ────────────────────────────────────────────────────

    def commit_block(
        self,
        start_pos:    int,
        end_pos:      int,
        layer_keys:   np.ndarray,   # (n_layers, n_kv_heads, block_size, head_dim)
        layer_values: np.ndarray,   # (n_layers, n_kv_heads, block_size, head_dim)
    ) -> bool:
        """
        Compress a complete block of KV data into engram memory.

        Returns True if committed, False if skipped (duplicate / empty).

        ✅ FIX-THREAD-SAFETY-COMMITTED-END: _last_committed_end is now updated
        inside _add_block() (while the lock is held), so callers that read
        _last_committed_end always see a value consistent with the block list.
        """
        if end_pos <= self._last_committed_end:
            return False
        block_len = end_pos - start_pos
        if block_len <= 0:
            return False

        n_l, n_h, n_d = self.n_layers, self.n_kv_heads, self.head_dim
        layer_keys_mean   = np.zeros((n_l, n_h, n_d), dtype=np.float32)
        layer_keys_max    = np.zeros((n_l, n_h, n_d), dtype=np.float32)
        layer_values_mean = np.zeros((n_l, n_h, n_d), dtype=np.float32)

        K = layer_keys.astype(np.float32)
        V = layer_values.astype(np.float32)

        for li in range(n_l):
            K_sel = K[li]
            V_sel = V[li]

            layer_keys_mean[li]   = K_sel.mean(axis=1)
            layer_values_mean[li] = V_sel.mean(axis=1)

            abs_max_idx = np.abs(K_sel).argmax(axis=1)
            h_idx = np.arange(n_h)[:, None]
            d_idx = np.arange(n_d)[None, :]
            layer_keys_max[li] = K_sel[h_idx, abs_max_idx, d_idx]

        raw_mean    = layer_keys_mean.mean(axis=0).mean(axis=0)
        raw_max     = layer_keys_max.mean(axis=0).mean(axis=0)
        raw_summary = (raw_mean + self.max_pool_weight * raw_max) @ self._W_summary
        summary     = _l2_norm(raw_summary.astype(np.float32))
        importance  = float(np.abs(layer_keys_mean).mean())

        blk = EngramBlock(
            start_pos         = start_pos,
            end_pos           = end_pos,
            layer_keys_mean   = layer_keys_mean.astype(self.dtype),
            layer_keys_max    = layer_keys_max.astype(self.dtype),
            layer_values_mean = layer_values_mean.astype(self.dtype),
            summary           = summary,
            importance        = importance,
        )
        # ✅ FIX-THREAD-SAFETY-COMMITTED-END: pass end_pos so _add_block can
        # update _last_committed_end atomically inside the same lock.
        self._add_block(blk, committed_end=end_pos)
        return True

    # ── Legacy compatibility shim ─────────────────────────────────────────────

    def maybe_commit(
        self,
        kv_cache: "KVCache",
        current_pos: int,
        force: bool = False,
    ) -> int:
        """
        Legacy commit interface — safe fallback using temporal-offset reads.
        Returns number of new blocks committed.
        """
        n_new = 0
        block_start = self._last_committed_end
        while block_start + self.block_size <= current_pos:
            block_end = block_start + self.block_size

            window            = kv_cache.window
            oldest_accessible = max(0, current_pos - window)
            if block_start < oldest_accessible:
                block_start = block_end
                # Advance pointer atomically inside lock
                with self._lock:
                    self._last_committed_end = block_end
                continue

            blk = self._extract_block_from_cache(
                kv_cache, block_start, block_end, current_pos)
            if blk is not None:
                # ✅ FIX-THREAD-SAFETY-COMMITTED-END: pass committed_end so
                # _add_block updates _last_committed_end inside the lock.
                self._add_block(blk, committed_end=block_end)
                n_new += 1
            block_start = block_end

        return n_new

    def _extract_block_from_cache(
        self,
        kv_cache: "KVCache",
        start_pos:   int,
        end_pos:     int,
        current_pos: int,
    ) -> Optional[EngramBlock]:
        """Extract and compress KV for [start_pos, end_pos) using temporal offset."""
        block_len = end_pos - start_pos
        if block_len <= 0:
            return None

        n_l, n_h, n_d = self.n_layers, self.n_kv_heads, self.head_dim
        layer_keys_mean   = np.zeros((n_l, n_h, n_d), dtype=np.float32)
        layer_keys_max    = np.zeros((n_l, n_h, n_d), dtype=np.float32)
        layer_values_mean = np.zeros((n_l, n_h, n_d), dtype=np.float32)

        for li in range(n_l):
            K_full, V_full = kv_cache.get(li)
            total_len  = K_full.shape[2]
            oldest_pos = current_pos - total_len

            if start_pos < oldest_pos:
                return None

            offset     = start_pos - oldest_pos
            end_offset = offset + block_len
            if end_offset > total_len:
                return None

            K_sel = K_full[0, :, offset:end_offset, :].astype(np.float32)
            V_sel = V_full[0, :, offset:end_offset, :].astype(np.float32)

            layer_keys_mean[li]   = K_sel.mean(axis=1)
            layer_values_mean[li] = V_sel.mean(axis=1)

            abs_max_idx = np.abs(K_sel).argmax(axis=1)
            h_idx = np.arange(n_h)[:, None]
            d_idx = np.arange(n_d)[None, :]
            layer_keys_max[li] = K_sel[h_idx, abs_max_idx, d_idx]

        raw_mean    = layer_keys_mean.mean(axis=0).mean(axis=0)
        raw_max     = layer_keys_max.mean(axis=0).mean(axis=0)
        raw_summary = (raw_mean + self.max_pool_weight * raw_max) @ self._W_summary
        summary     = _l2_norm(raw_summary.astype(np.float32))
        importance  = float(np.abs(layer_keys_mean).mean())

        return EngramBlock(
            start_pos         = start_pos,
            end_pos           = end_pos,
            layer_keys_mean   = layer_keys_mean.astype(self.dtype),
            layer_keys_max    = layer_keys_max.astype(self.dtype),
            layer_values_mean = layer_values_mean.astype(self.dtype),
            summary           = summary,
            importance        = importance,
        )

    def _add_block(self, blk: EngramBlock, committed_end: Optional[int] = None):
        """
        Thread-safe insertion with lazy matrix invalidation.

        ✅ FIX-THREAD-SAFETY-COMMITTED-END: committed_end, when provided, is
        written to _last_committed_end inside the lock — atomically with the
        block append.
        """
        with self._lock:
            if self.max_blocks is not None and len(self._blocks) >= self.max_blocks:
                self._blocks.popleft()
                self._matrix_dirty  = True
                self._toc_summaries = None
                self._toc_block_idx = []

            self._blocks.append(blk)
            self._n_commits   += 1
            self._matrix_dirty = True

            # ✅ atomic update of committed pointer
            if committed_end is not None:
                self._last_committed_end = committed_end

            if len(self._blocks) % self.n_toc_blocks == 0:
                self._rebuild_toc_nolock()

    # ── Matrix / ToC rebuild ──────────────────────────────────────────────────

    def _rebuild_summary_matrix_nolock(self):
        """Rebuild pre-allocated scoring matrix (call with _lock held)."""
        n = len(self._blocks)
        if n == 0:
            self._summary_matrix = None
            self._importance_vec = None
        else:
            self._summary_matrix = np.stack(
                [b.summary for b in self._blocks], axis=0
            ).astype(np.float32)
            self._importance_vec = np.array(
                [b.importance for b in self._blocks], dtype=np.float32
            )
        self._matrix_dirty = False

    def _rebuild_toc_nolock(self):
        """Rebuild table-of-contents (call with _lock held)."""
        n_blocks = len(self._blocks)
        if n_blocks == 0:
            self._toc_summaries = None
            self._toc_block_idx = []
            self._matrix_dirty  = True
            return

        n_chapters = math.ceil(n_blocks / self.n_toc_blocks)
        toc        = np.zeros((n_chapters, self.head_dim), dtype=np.float32)
        idx_ranges: List[Tuple[int, int]] = []

        for ch in range(n_chapters):
            b0 = ch * self.n_toc_blocks
            b1 = min(b0 + self.n_toc_blocks, n_blocks)
            sums = np.stack(
                [self._blocks[b].summary for b in range(b0, b1)], axis=0
            )
            toc[ch] = _l2_norm(sums.mean(axis=0))
            idx_ranges.append((b0, b1))

        self._toc_summaries = toc
        self._toc_block_idx = idx_ranges
        self._matrix_dirty  = True

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve_for_layer(
        self,
        layer_idx: int,
        query_vec: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Top-k retrieval for one layer."""
        with self._lock:
            n_blocks = len(self._blocks)
            if n_blocks == 0:
                return None, None, None

            self._n_retrieves += 1

            if self._matrix_dirty:
                self._rebuild_summary_matrix_nolock()

            candidates = self._toc_filter(query_vec, n_blocks)
            selected   = self._score_blocks_vectorized(query_vec, candidates)
            if not selected:
                return None, None, None

            k_list = [self._blocks[i].layer_keys_mean[layer_idx]   for i in selected]
            v_list = [self._blocks[i].layer_values_mean[layer_idx] for i in selected]
            K_stack = np.stack(k_list, axis=1)
            V_stack = np.stack(v_list, axis=1)
            eng_K   = K_stack[None].astype(np.float32)
            eng_V   = V_stack[None].astype(np.float32)

            return eng_K, eng_V, None

    def _toc_filter(self, query_vec: np.ndarray, n_blocks: int) -> List[int]:
        """Two-level filter: ToC cosine similarity → candidate block indices."""
        if self._toc_summaries is None or not self._toc_block_idx:
            limit = min(n_blocks, self.top_k_retrieve * 4)
            return list(range(n_blocks - limit, n_blocks))

        q_sum  = _l2_norm(query_vec.mean(axis=0).astype(np.float32))
        scores = self._toc_summaries @ q_sum

        n_need = max(1, math.ceil(self.top_k_retrieve / self.n_toc_blocks))
        n_ch   = len(self._toc_block_idx)
        n_pick = min(n_need, n_ch)

        top_ch = (np.argpartition(scores, -n_pick)[-n_pick:]
                  if n_pick < n_ch else np.arange(n_ch))

        candidates: List[int] = []
        for ch in top_ch.tolist():
            b0, b1 = self._toc_block_idx[ch]
            candidates.extend(range(b0, b1))
        return candidates

    def _score_blocks_vectorized(
        self,
        query_vec: np.ndarray,
        candidate_indices: List[int],
    ) -> List[int]:
        """Pure NumPy BLAS dot-product over all candidates; argpartition O(n)."""
        if not candidate_indices or self._summary_matrix is None:
            return []

        q_sum = _l2_norm(query_vec.mean(axis=0).astype(np.float32))

        cand  = np.array(candidate_indices, dtype=np.int32)
        sums  = self._summary_matrix[cand]
        raw_imps = self._importance_vec[cand]

        imps_norm = 1.0 / (1.0 + np.exp(-np.clip(raw_imps, -20.0, 20.0)))

        scores = sums @ q_sum
        scores = scores * (1.0 + 0.1 * imps_norm)

        k = min(self.top_k_retrieve, len(candidate_indices))
        if k < len(scores):
            top_local = np.argpartition(scores, -k)[-k:]
        else:
            top_local = np.arange(len(scores))

        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return [candidate_indices[int(i)] for i in top_local]

    # ── Engram cross-attention ─────────────────────────────────────────────────

    def attend(
        self,
        layer_idx:   int,
        Q:           np.ndarray,
        n_rep:       int,
        scale:       float,
        softmax_eps: float = 1e-9,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Cross-attention of Q over engram KV for this layer.

        ✅ FIX-UNUSED-VAR-SEQ: removed `seq = Q.shape[2]` which was assigned
        but never used in the method body.  The query mean is computed
        directly from Q[0] and Q[0, :, :, :].mean(axis=1).
        """
        query_vec = Q[0, :, :, :].mean(axis=1)
        kv_query  = query_vec.reshape(
            self.n_kv_heads, n_rep, self.head_dim).mean(axis=1)

        eng_K, eng_V, _ = self.retrieve_for_layer(layer_idx, kv_query)
        if eng_K is None:
            return None, self.blend_alpha

        eng_K_exp = np.repeat(eng_K, n_rep, axis=1)
        eng_V_exp = np.repeat(eng_V, n_rep, axis=1)

        eng_attn = Q @ eng_K_exp.transpose(0, 1, 3, 2) * scale
        eng_attn = np.clip(eng_attn, -50.0, 50.0)
        eng_attn -= eng_attn.max(axis=-1, keepdims=True)
        eng_w     = np.exp(eng_attn)
        eng_w    /= eng_w.sum(axis=-1, keepdims=True) + softmax_eps
        eng_out   = eng_w @ eng_V_exp

        eff_alpha = self.blend_alpha
        if self.dynamic_alpha:
            max_sim = float(eng_w[0, :, -1, :].max())
            lo, hi  = self.alpha_low_thresh, self.alpha_high_thresh
            if max_sim > lo:
                t         = min(1.0, (max_sim - lo) / (hi - lo + 1e-9))
                eff_alpha = self.blend_alpha - 0.30 * t
                eff_alpha = max(0.50, eff_alpha)

        return eng_out, eff_alpha

    # ── Snapshot / restore ────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Lightweight snapshot for speculative rollback."""
        with self._lock:
            return {
                "_n_blocks":           len(self._blocks),
                "_last_committed_end": self._last_committed_end,
            }

    def restore(self, snap: dict):
        """Restore to a previous snapshot (trim extra blocks)."""
        with self._lock:
            n_target = snap["_n_blocks"]
            if len(self._blocks) > n_target:
                while len(self._blocks) > n_target:
                    self._blocks.pop()
                self._matrix_dirty  = True
                self._toc_summaries = None
                self._toc_block_idx = []
                self._rebuild_toc_nolock()
            self._last_committed_end = snap["_last_committed_end"]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        import os, json
        os.makedirs(path, exist_ok=True)
        with self._lock:
            meta = {
                "n_layers":           self.n_layers,
                "n_kv_heads":         self.n_kv_heads,
                "head_dim":           self.head_dim,
                "d_model":            self.d_model,
                "block_size":         self.block_size,
                "top_k_retrieve":     self.top_k_retrieve,
                "n_toc_blocks":       self.n_toc_blocks,
                "blend_alpha":        self.blend_alpha,
                "dynamic_alpha":      self.dynamic_alpha,
                "alpha_high_thresh":  self.alpha_high_thresh,
                "alpha_low_thresh":   self.alpha_low_thresh,
                "max_pool_weight":    self.max_pool_weight,
                "last_committed_end": self._last_committed_end,
                "n_blocks":           len(self._blocks),
            }
            np.save(f"{path}/W_summary.npy", self._W_summary)
            with open(f"{path}/meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            for i, blk in enumerate(self._blocks):
                np.save(f"{path}/block_{i}_Km.npy",  blk.layer_keys_mean)
                np.save(f"{path}/block_{i}_Kx.npy",  blk.layer_keys_max)
                np.save(f"{path}/block_{i}_V.npy",   blk.layer_values_mean)
                np.save(f"{path}/block_{i}_sum.npy", blk.summary)
                np.save(f"{path}/block_{i}_meta.npy",
                        np.array([blk.start_pos, blk.end_pos,
                                  np.float32(blk.importance)]))

    @classmethod
    def load(cls, path: str) -> "EngramCache":
        import json
        with open(f"{path}/meta.json") as f:
            meta = json.load(f)

        eng = cls(
            n_layers          = meta["n_layers"],
            n_kv_heads        = meta["n_kv_heads"],
            head_dim          = meta["head_dim"],
            d_model           = meta["d_model"],
            block_size        = meta["block_size"],
            top_k_retrieve    = meta["top_k_retrieve"],
            n_toc_blocks      = meta["n_toc_blocks"],
            blend_alpha       = meta["blend_alpha"],
            dynamic_alpha     = meta.get("dynamic_alpha", True),
            alpha_high_thresh = meta.get("alpha_high_thresh", 0.85),
            alpha_low_thresh  = meta.get("alpha_low_thresh", 0.50),
            max_pool_weight   = meta.get("max_pool_weight", 0.20),
        )
        eng._W_summary = np.load(f"{path}/W_summary.npy")
        eng._last_committed_end = meta.get(
            "last_committed_end", meta.get("committed_up_to", 0))

        for i in range(meta["n_blocks"]):
            Km = np.load(f"{path}/block_{i}_Km.npy")
            Kx = np.load(f"{path}/block_{i}_Kx.npy")
            V  = np.load(f"{path}/block_{i}_V.npy")
            s  = np.load(f"{path}/block_{i}_sum.npy")
            bm = np.load(f"{path}/block_{i}_meta.npy")
            eng._blocks.append(EngramBlock(
                start_pos         = int(bm[0]),
                end_pos           = int(bm[1]),
                layer_keys_mean   = Km,
                layer_keys_max    = Kx,
                layer_values_mean = V,
                summary           = s,
                importance        = float(bm[2]),
            ))

        with eng._lock:
            eng._matrix_dirty = True
            eng._rebuild_toc_nolock()
        return eng

    # ── Utilities ─────────────────────────────────────────────────────────────

    def clear(self):
        with self._lock:
            self._blocks.clear()
            self._summary_matrix     = None
            self._importance_vec     = None
            self._matrix_dirty       = True
            self._toc_summaries      = None
            self._toc_block_idx      = []
            self._last_committed_end = 0
            self._n_commits          = 0
            self._n_retrieves        = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_blocks":            len(self._blocks),
                "last_committed_end":  self._last_committed_end,
                "n_toc_chapters":      len(self._toc_block_idx),
                "n_commits":           self._n_commits,
                "n_retrieves":         self._n_retrieves,
                "compression_ratio":   self.block_size,
                "tokens_in_engram":    len(self._blocks) * self.block_size,
                "dynamic_alpha":       self.dynamic_alpha,
                "blend_alpha_base":    self.blend_alpha,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._blocks)

    def __repr__(self) -> str:
        s = self.stats()
        dyn = "→dynamic" if self.dynamic_alpha else ""
        return (
            f"EngramCache(blocks={s['n_blocks']}, "
            f"tokens={s['tokens_in_engram']}, "
            f"last_committed_end={s['last_committed_end']}, "
            f"blend_alpha={self.blend_alpha}{dyn}, "
            f"block_size={self.block_size})"
        )