# DracoAI V1 — modeling/runtime/speculative.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Speculative Decoding — MTPHead, SpeculativeDecoder, SpeculativeTreeDecoder.

FIXES (this revision):
  ✅ FIX-TREE-L2-STALE         : _search() returns l2 from the LAST replayed
     forward call on the winning chain.
  ✅ FIX-TREE-VERIFY-CORRECT   : verify_id is sampled from cur_logits (the
     verifier's distribution), compared against spec_id (the draft candidate).
  ✅ FIX-TREE-REPLAY-L2        : winning chain replay captures final l2 from
     the last model.forward() so the caller gets a fresh MTP state.
  ✅ FIX-TREE-EOS-KV           : when an EOS candidate is accepted in _search,
     model.forward() is still called so the KV for that token is written into
     the cache before generation ends.
  ✅ FIX-TREE-MU-PROPAGATION   : verify_mu is propagated correctly when an EOS
     candidate is found so the caller's mu state is consistent.
  ✅ KEEP FIX-SPEC-AUTO-DISABLE-STICKY (EMA hysteresis).
  ✅ FIX-TREE-EOS-RESTORE-BUG  : removed cache.restore(branch_snap) on EOS
     accept.  Restoring after EOS accept wiped the EOS token's KV from the
     cache, creating a KV/sequence desync on any subsequent continued
     generation.
  ✅ FIX-TREE-BUDGET-REPLAY    : try_tree now accepts a max_tokens budget; the
     winning chain is truncated to that budget BEFORE replay so the KV cache
     never ends up with more entries than the generate loop will commit to ids.
  ✅ FIX-TREE-LOGITS-STALE-AFTER-TRUNCATION : when the winning chain is
     truncated to max_tokens, best_logits is updated to the logits from the
     LAST actually-replayed token (not the pre-truncation end of chain).
  ✅ FIX-UNUSED-VAR-WAS-TRUNCATED : removed the local variable `was_truncated`
     that was assigned but never read.  The truncation condition only affects
     `best_accepted`; no downstream code branches on this flag.
"""
from __future__ import annotations
import math
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..constants import SPEC_THRESH, DEFAULT_TEMP, DEFAULT_TOP_P
from ..sampling.sampler import Sampler

if TYPE_CHECKING:
    from ..transformer import DracoTransformerV1
    from ..kv_cache.kv_cache import KVCache

__all__ = ["MTPHead", "SpeculativeDecoder", "SpeculativeTreeDecoder"]


class MTPHead:
    def __init__(self, d_model: int, vocab_size: int):
        scale   = 1.0 / math.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W2 = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.lm_head: Optional[np.ndarray] = None
        self.d_model  = d_model

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.lm_head is None:
            raise RuntimeError("MTPHead.lm_head is None — assign before calling forward().")

        def _silu(z: np.ndarray) -> np.ndarray:
            return z / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

        h1 = _silu(x @ self.W1)
        h2 = _silu(h1 @ self.W2)
        W  = self.lm_head.astype(np.float32) if self.lm_head.dtype != np.float32 else self.lm_head
        return h1 @ W.T, h2 @ W.T

    def try_speculative(self, l2: np.ndarray,
                        thresh: float = SPEC_THRESH) -> Tuple[Optional[int], float]:
        last = l2[0, -1].astype(np.float64)
        last = np.clip(last, -50, 50)
        probs = np.exp(last - last.max())
        probs /= probs.sum() + 1e-9
        best_id   = int(probs.argmax())
        best_prob = float(probs[best_id])
        return (best_id, best_prob) if best_prob >= thresh else (None, 0.0)

    def try_speculative_topk(self, l2: np.ndarray, thresh: float = SPEC_THRESH,
                             top_k_beam: int = 3) -> List[Tuple[int, float]]:
        last = l2[0, -1].astype(np.float64)
        last = np.clip(last, -50, 50)
        probs = np.exp(last - last.max())
        probs /= probs.sum() + 1e-9
        top_ids = (np.argsort(probs)[::-1] if top_k_beam >= len(probs)
                   else np.argpartition(probs, -top_k_beam)[-top_k_beam:])
        top_ids = top_ids[np.argsort(probs[top_ids])[::-1]]
        return [(int(i), float(probs[i])) for i in top_ids if float(probs[i]) >= thresh]


class SpeculativeDecoder:
    """EMA accept-rate tracker with hysteresis auto-disable/re-enable."""

    def __init__(self, disable_thresh: float = 0.3, ema_alpha: float = 0.1,
                 re_enable_factor: float = 1.5):
        self._disable_thresh   = disable_thresh
        self._re_enable_thresh = disable_thresh * re_enable_factor
        self._ema_alpha        = ema_alpha
        self._accept_ema: float = 1.0
        self._n_events:   int   = 0
        self.suggest_disable: bool = False

    def record_accept(self):
        self._n_events += 1
        self._accept_ema = self._ema_alpha + (1 - self._ema_alpha) * self._accept_ema
        self._update_suggestion()

    def record_reject(self):
        self._n_events += 1
        self._accept_ema = (1 - self._ema_alpha) * self._accept_ema
        self._update_suggestion()

    def _update_suggestion(self):
        if self._n_events < 10:
            return
        if not self.suggest_disable:
            if self._accept_ema < self._disable_thresh:
                self.suggest_disable = True
        else:
            if self._accept_ema > self._re_enable_thresh:
                self.suggest_disable = False

    def reset(self):
        self._accept_ema = 1.0; self._n_events = 0; self.suggest_disable = False

    @property
    def accept_rate(self) -> float:
        return self._accept_ema

    def __repr__(self) -> str:
        return (f"SpeculativeDecoder(accept_ema={self._accept_ema:.2f}, "
                f"events={self._n_events}, suggest_disable={self.suggest_disable})")


class SpeculativeTreeDecoder:
    """
    Speculative Tree Decoding.

    Invariants:
    ─────────────────────────────────────────────────────────────────────
    • model.forward() is called for EVERY accepted token (including EOS)
      so the KV cache is always consistent with the accepted sequence.
    • verify_id is sampled from the VERIFIER's cur_logits distribution,
      then compared to the DRAFT spec_id.
    • Winning chain replay captures the l2 from the final forward() call
      so the caller receives a fresh MTP state.
    • Branch snapshots are always restored after each candidate exploration,
      whether the candidate was accepted or rejected.
    • EOS accept does NOT restore branch_snap — KV of EOS is kept in cache.
    • Winning chain is truncated to max_tokens budget BEFORE replay so the
      KV cache entry count always matches what the generate loop commits.
    • best_logits is updated during replay to reflect the logits of the last
      actually-replayed token (FIX-TREE-LOGITS-STALE-AFTER-TRUNCATION).
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, model: "DracoTransformerV1", tree_width: int = 3,
                 tree_depth: int = 2, thresh: float = SPEC_THRESH):
        self.model      = model
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.thresh     = thresh

    def try_tree(
        self,
        cache:        "KVCache",
        logits:       np.ndarray,
        l2:           np.ndarray,
        ids:          List[int],
        freq:         dict,
        pos:          dict,
        n_pos:        int,
        _eos_set:     set,
        mu:           float,
        use_mirostat: bool,
        temp:         float = DEFAULT_TEMP,
        top_p:        float = DEFAULT_TOP_P,
        min_p:        float = 0.0,
        tau:          float = 5.0,
        eta:          float = 0.1,
        intent_boost: Optional[np.ndarray] = None,
        intent_bias:  Optional[np.ndarray] = None,
        add_noise:    bool  = True,
        max_tokens:   int   = 0,
    ) -> Tuple[List[int], np.ndarray, np.ndarray, float]:
        """
        Returns (accepted_ids, final_logits, final_l2, final_mu).

        accepted_ids is truncated to max_tokens (if > 0) BEFORE replay so
        the KV cache is never ahead of what the generate loop actually commits
        to ids.  final_l2 reflects the post-replay (truncated) cache state.
        final_logits is from the LAST replayed token — guarantees freshness
        whether or not truncation occurred (FIX-TREE-LOGITS-STALE-AFTER-TRUNCATION).
        If nothing is accepted, cache state is restored to root_snap.
        """
        model = self.model

        def _sample(lg: np.ndarray, _mu: float) -> Tuple[int, float]:
            if use_mirostat:
                return Sampler.mirostat_v2(lg, _mu, tau, eta)
            return Sampler.topk_topp(lg, temp, top_p, min_p=min_p), _mu

        # Take a full root snapshot; restore if nothing accepted.
        root_snap = cache.snapshot(delta_threshold=0)
        cache._snap_escalate_to_full(root_snap)

        def _search(
            cur_l2:     np.ndarray,
            cur_logits: np.ndarray,
            depth:      int,
            cur_mu:     float,
        ) -> Tuple[List[int], np.ndarray, np.ndarray, float]:
            """
            Recursively search tree candidates.
            Returns (chain, final_logits, final_l2, final_mu).
            Cache state on return: restored to the state at _search entry,
            EXCEPT when an EOS token is accepted (cache keeps EOS KV).
            """
            if depth == 0:
                return [], cur_logits, cur_l2, cur_mu

            candidates = model.mtp.try_speculative_topk(cur_l2, self.thresh, self.tree_width)
            if not candidates:
                return [], cur_logits, cur_l2, cur_mu

            best_accepted: List[int] = []
            best_logits              = cur_logits
            best_l2                  = cur_l2
            best_mu                  = cur_mu

            for spec_id, _ in candidates:
                # Take branch snapshot BEFORE forwarding spec_id.
                branch_snap = cache.snapshot(delta_threshold=0)
                cache._snap_escalate_to_full(branch_snap)

                # Always forward spec_id so its KV is written into cache.
                l1_c, l2_c, _ = model.forward(
                    [spec_id], cache,
                    intent_boost=intent_boost,
                    add_noise=add_noise,
                    intent_bias=intent_bias,
                )
                branch_logits = np.clip(l1_c[0, -1].astype(np.float64), -50.0, 50.0)

                # Sample from VERIFIER's distribution (cur_logits) and accept
                # only if it matches the draft candidate.
                verify_id, verify_mu = _sample(cur_logits.copy(), cur_mu)

                if verify_id == spec_id:
                    if spec_id in _eos_set:
                        # ✅ FIX-TREE-EOS-RESTORE-BUG: do NOT restore the
                        # branch snapshot here.  model.forward() already
                        # wrote the EOS token's KV into the cache.
                        # Restoring would erase that entry and leave the
                        # cache one position behind the sequence, corrupting
                        # any continued generation.  We keep the cache as-is
                        # and return — generation ends immediately after.
                        return [spec_id], branch_logits, l2_c, verify_mu

                    deeper, d_logits, d_l2, d_mu = _search(
                        l2_c, branch_logits, depth - 1, verify_mu)
                    chain = [spec_id] + deeper
                    if len(chain) > len(best_accepted):
                        best_accepted = chain
                        best_logits   = d_logits
                        best_l2       = d_l2
                        best_mu       = d_mu

                # Restore cache to the state before this branch was explored.
                cache.restore(branch_snap)

            # ✅ FIX-TREE-BUDGET-REPLAY: truncate the winning chain to the
            # caller's remaining token budget BEFORE replaying into cache.
            # Without this, cache ends up with more KV entries than ids,
            # causing phantom attention positions on the next forward pass.
            # ✅ FIX-UNUSED-VAR-WAS-TRUNCATED: the boolean flag is not needed
            # — truncation only affects best_accepted in-place; downstream
            # replay always uses best_logits from the last replayed token.
            if max_tokens > 0 and len(best_accepted) > max_tokens:
                best_accepted = best_accepted[:max_tokens]

            # Replay the (possibly truncated) winning chain on the restored
            # cache so KV exactly matches the committed token sequence.
            # ✅ FIX-TREE-LOGITS-STALE-AFTER-TRUNCATION: capture l1 from the
            # replay loop so best_logits reflects the LAST replayed token.
            # When truncation occurred, the pre-truncation best_logits was
            # from beyond the cut-off point; the caller must receive logits
            # from the actual last committed token for correct next-step
            # sampling.
            if best_accepted:
                replay_l2     = best_l2
                replay_logits = best_logits
                for tok in best_accepted:
                    l1_r, replay_l2, _ = model.forward(
                        [tok], cache,
                        intent_boost=intent_boost,
                        add_noise=add_noise,
                        intent_bias=intent_bias,
                    )
                    replay_logits = np.clip(l1_r[0, -1].astype(np.float64), -50.0, 50.0)
                best_l2 = replay_l2  # fresh MTP state after replay
                # Always update best_logits from replay (guarantees freshness
                # whether or not truncation occurred, at negligible extra cost).
                best_logits = replay_logits

            return best_accepted, best_logits, best_l2, best_mu

        result       = _search(l2, logits, self.tree_depth, mu)
        accepted_ids = result[0]

        if not accepted_ids:
            # Nothing accepted — restore to root so caller is in clean state.
            cache.restore(root_snap)

        return result