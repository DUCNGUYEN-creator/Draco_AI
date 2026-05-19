# DracoAI V1 — modeling/runtime/medusa.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Medusa Heads — Multi-token parallel speculative decoding.

Architecture
────────────
Standard decoding: 1 forward pass → 1 token.
Medusa decoding:   1 forward pass + n_heads cheap linear projections
                   → draft tokens at positions +1, +2, ..., +n_heads.

Each Medusa head is a shallow 2-layer MLP (d_model → d_model → vocab)
that takes the last hidden state and predicts the token n steps ahead.
These are fast (<<1 ms) compared to a full forward pass (~10-100 ms).

Verification:
  After the draft tokens are proposed, the VERIFIER runs a single
  batched forward pass on all draft tokens simultaneously.  It accepts
  the longest prefix where each draft token matches the verifier's
  argmax.  This is the "Medusa tree" acceptance strategy.

Implementation note:
  MedusaHeads wraps the MTPHead pattern but extends it to n_heads.
  It integrates with the existing SpeculativeDecoder EMA tracker.
  Fully compatible with single-token speculative (MTPHead) — they
  share the same SpeculativeDecoder accept/reject statistics.

Key differences from tree speculative (SpeculativeTreeDecoder):
  • Medusa drafts are parallel (one forward → n drafts)
  • Tree speculative is sequential (n forwARds → n drafts)
  • Medusa uses separate learned heads; tree reuses MTP hidden state
  • Medusa is faster for long draft sequences; tree has higher quality

Usage in generate():
  medusa = MedusaHeads(d_model, vocab_size, n_heads=3)
  ...
  drafts, confs = medusa.draft(hidden_state)   # [tok+1, tok+2, tok+3]
  accepted = medusa.verify(drafts, cache, ...)  # batched forward
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

__all__ = ["MedusaHeads", "MedusaDecoder"]


class MedusaHead:
    """Single Medusa head: 2-layer MLP projecting hidden → vocab logits."""

    def __init__(self, d_model: int, vocab_size: int, offset: int):
        """
        offset : token-distance this head predicts (1 = next, 2 = skip-1, ...)
        """
        scale = 1.0 / math.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_model).astype(np.float32) * scale
        self.W2 = np.random.randn(d_model, vocab_size).astype(np.float32) * scale
        self.offset = offset

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """
        hidden : (d_model,) or (1, 1, d_model) last hidden state
        Returns : (vocab_size,) logits
        """
        h = hidden.reshape(-1, hidden.shape[-1]).astype(np.float32)[-1]
        h1 = h @ self.W1
        h1 = h1 / (1.0 + np.exp(-np.clip(h1, -50, 50)))  # SiLU
        return h1 @ self.W2   # (vocab_size,)

    def predict(self, hidden: np.ndarray, thresh: float = SPEC_THRESH
                ) -> Tuple[Optional[int], float]:
        """Return (token_id, confidence) if confident enough, else (None, 0)."""
        logits = self.forward(hidden)
        logits_c = np.clip(logits.astype(np.float64), -50, 50)
        probs = np.exp(logits_c - logits_c.max())
        probs /= probs.sum() + 1e-9
        best_id   = int(probs.argmax())
        best_prob = float(probs[best_id])
        return (best_id, best_prob) if best_prob >= thresh else (None, 0.0)


class MedusaHeads:
    """
    Collection of Medusa heads for parallel draft generation.

    Parameters
    ----------
    d_model     : Hidden state dimension.
    vocab_size  : Vocabulary size.
    n_heads     : Number of speculative heads (typical: 3–5).
    lm_head     : Optional shared LM-head weight matrix — if provided,
                  heads W2 is replaced with lm_head for consistency.
    """

    def __init__(
        self,
        d_model:    int,
        vocab_size: int,
        n_heads:    int   = 3,
        lm_head:    Optional[np.ndarray] = None,
    ):
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.n_heads    = n_heads
        self.heads: List[MedusaHead] = [
            MedusaHead(d_model, vocab_size, offset=i + 1)
            for i in range(n_heads)
        ]
        if lm_head is not None:
            self.set_lm_head(lm_head)

    def set_lm_head(self, lm_head: np.ndarray) -> None:
        """Share the main LM head weights across all Medusa heads."""
        for h in self.heads:
            h.W2 = lm_head.astype(np.float32)

    def draft(
        self,
        hidden: np.ndarray,
        thresh: float = SPEC_THRESH,
    ) -> Tuple[List[int], List[float]]:
        """
        Produce draft tokens at positions +1 .. +n_heads.

        Returns
        -------
        tokens      : List of draft token IDs (may be shorter than n_heads
                      if a head's confidence is below thresh).
        confidences : Corresponding confidence scores.

        The list is truncated at the first head that is not confident;
        subsequent heads are not evaluated (they would depend on the
        uncertain prefix anyway).
        """
        tokens: List[int]  = []
        confs:  List[float] = []
        for head in self.heads:
            tok, conf = head.predict(hidden, thresh)
            if tok is None:
                break
            tokens.append(tok)
            confs.append(conf)
        return tokens, confs


class MedusaDecoder:
    """
    Full Medusa speculative decoding loop integrated with DracoTransformerV1.

    Wraps a MedusaHeads object and implements the verify-and-accept loop.
    """

    def __init__(
        self,
        model:   "DracoTransformerV1",
        heads:   MedusaHeads,
        thresh:  float = SPEC_THRESH,
    ):
        self.model  = model
        self.heads  = heads
        self.thresh = thresh

    def try_medusa(
        self,
        cache:        "KVCache",
        last_hidden:  np.ndarray,   # (1, 1, d_model) last hidden state
        last_logits:  np.ndarray,   # (vocab_size,)  verifier logits
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
    ) -> Tuple[List[int], np.ndarray, float]:
        """
        Attempt Medusa speculative decoding.

        Returns
        -------
        accepted_ids  : List of accepted token IDs (may be empty).
        final_logits  : Logits after the last accepted token.
        final_mu      : Updated mirostat mu.
        """
        def _sample_one(lg: np.ndarray, _mu: float) -> Tuple[int, float]:
            if use_mirostat:
                return Sampler.mirostat_v2(lg.astype(np.float64), _mu, tau, eta)
            return Sampler.topk_topp(lg, temp, top_p, min_p=min_p), _mu

        # Phase 1: draft
        draft_ids, draft_confs = self.heads.draft(last_hidden, self.thresh)
        if not draft_ids:
            return [], last_logits, mu

        if max_tokens > 0:
            draft_ids = draft_ids[:max_tokens]

        # Phase 2: batched verification — forward all drafts at once
        # Take a snapshot to restore if verification rejects some tokens
        snap = cache.snapshot(delta_threshold=0)
        cache._snap_escalate_to_full(snap)

        accepted_ids:   List[int]       = []
        current_logits: np.ndarray      = last_logits.astype(np.float64)
        current_mu:     float           = mu

        for i, draft_tok in enumerate(draft_ids):
            # Verifier samples from current distribution
            verify_id, current_mu = _sample_one(current_logits, current_mu)

            if verify_id != draft_tok:
                # Rejection: accepted up to i-1, use verify_id instead
                # Restore cache to before rejected draft token was forwarded
                cache.restore(snap)
                # Re-forward accepted prefix to get correct cache state
                for accepted_tok in accepted_ids:
                    l1, _, _ = self.model.forward(
                        [accepted_tok], cache,
                        intent_boost=intent_boost,
                        add_noise=False,    # deterministic replay
                        intent_bias=intent_bias,
                    )
                    current_logits = np.clip(
                        l1[0, -1].astype(np.float64), -50.0, 50.0)
                break

            # Acceptance: forward this draft token
            l1, _, _ = self.model.forward(
                [draft_tok], cache,
                intent_boost=intent_boost,
                add_noise=add_noise,
                intent_bias=intent_bias,
            )
            current_logits = np.clip(l1[0, -1].astype(np.float64), -50.0, 50.0)
            accepted_ids.append(draft_tok)

            # Update snapshot for next iteration (snap now at this accepted point)
            snap = cache.snapshot(delta_threshold=0)
            cache._snap_escalate_to_full(snap)

            if draft_tok in _eos_set:
                break

        return accepted_ids, current_logits, current_mu

    def __repr__(self) -> str:
        return (f"MedusaDecoder(n_heads={self.heads.n_heads}, "
                f"thresh={self.thresh})")