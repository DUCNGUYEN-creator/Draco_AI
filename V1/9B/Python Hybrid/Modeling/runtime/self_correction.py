# DracoAI V1 — modeling/runtime/self_correction.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Self-Correction Loop — re-sample with diversity boost on low confidence.

When the model produces very low-confidence token distributions for
several consecutive steps (detected by HealthMonitor), the
SelfCorrectionManager recommends a re-sample with elevated temperature
and optionally suppresses already-seen tokens more aggressively.

Design principles
─────────────────
• NO extra forward passes.  Re-sampling is cheap (O(vocab) CPU) vs
  re-running the full transformer (O(n_layers × seq × d²) FLOPS).
• The existing `last_logits` are reused — the correction is in the
  SAMPLING distribution, not in the model's hidden state.
• Fully decoupled: SelfCorrectionManager is a stateless utility that
  generate() consults optionally.  If not used, zero overhead.
• Compatible with mirostat and top-k/top-p sampling.

Usage in generate() (pseudo-code):
    signal = health_monitor.check_step(last_logits)
    if signal.should_correct:
        last_logits = sc_manager.diversify_logits(last_logits, signal)
    nid, conf = _sample(last_logits)

The correction raises effective temperature and boosts rare tokens,
increasing the chance of escaping a low-confidence rut without making
the output chaotic.
"""
from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .health import SelfCorrectionSignal

__all__ = ["SelfCorrectionManager"]


class SelfCorrectionManager:
    """
    Manages diversity-boosted re-sampling when the model is uncertain.

    Parameters
    ----------
    temp_boost        : Multiplicative factor applied to logits (÷ temp_boost
                        = higher effective temperature).  1.5 is a safe default.
    rare_token_boost  : Flat additive bonus on logits for tokens below
                        `rare_threshold` probability.  Prevents the model from
                        always falling back to the same small safe vocabulary.
    rare_threshold    : Probability threshold below which a token is "rare".
    max_consecutive   : After this many corrections in a row, reset the
                        patience counter regardless (prevents infinite loop).
    freq_penalty_mult : When correcting, multiply the repetition penalty by
                        this factor to more aggressively suppress repetition.
    """

    def __init__(
        self,
        temp_boost:       float = 1.5,
        rare_token_boost: float = 0.3,
        rare_threshold:   float = 0.001,
        max_consecutive:  int   = 5,
        freq_penalty_mult: float = 2.0,
    ):
        self.temp_boost        = max(1.0, temp_boost)
        self.rare_token_boost  = rare_token_boost
        self.rare_threshold    = rare_threshold
        self.max_consecutive   = max_consecutive
        self.freq_penalty_mult = freq_penalty_mult

        self._n_corrections        = 0
        self._consecutive_correct  = 0

    # ── Core correction ───────────────────────────────────────────────────────

    def diversify_logits(
        self,
        logits: np.ndarray,
        signal: "SelfCorrectionSignal",
        freq:   Optional[dict] = None,
    ) -> np.ndarray:
        """
        Return modified logits with diversity-boosted distribution.

        Parameters
        ----------
        logits : (vocab_size,) float64 — raw logits from last forward.
        signal : SelfCorrectionSignal from HealthMonitor.check_step().
        freq   : Optional token frequency dict for boosted rep-penalty.

        Returns
        -------
        (vocab_size,) float64 — modified logits for re-sampling.
        """
        if not signal.should_correct:
            self._consecutive_correct = 0
            return logits

        # Guard: don't correct indefinitely
        if self._consecutive_correct >= self.max_consecutive:
            self._consecutive_correct = 0
            return logits

        self._n_corrections       += 1
        self._consecutive_correct += 1

        logits = logits.copy()

        # 1. Temperature boost: divide by temp_boost (flatter distribution)
        logits = logits / self.temp_boost

        # 2. Rare token boost: give low-probability tokens a floor
        if self.rare_token_boost > 0.0:
            lg_shifted = logits - logits.max()
            probs      = np.exp(lg_shifted)
            probs     /= probs.sum() + 1e-9
            rare_mask  = probs < self.rare_threshold
            logits[rare_mask] += self.rare_token_boost

        # 3. Boosted repetition penalty on frequent tokens
        if freq and self.freq_penalty_mult > 1.0:
            for tid, cnt in freq.items():
                if cnt > 1 and 0 <= tid < len(logits):
                    logits[tid] -= (
                        (self.freq_penalty_mult - 1.0)
                        * math.log(1 + cnt)
                    )

        return logits

    # ── Stats ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._n_corrections       = 0
        self._consecutive_correct = 0

    def stats(self) -> dict:
        return {
            "n_corrections":       self._n_corrections,
            "consecutive_correct": self._consecutive_correct,
            "temp_boost":          self.temp_boost,
        }

    def __repr__(self) -> str:
        return (f"SelfCorrectionManager(temp_boost={self.temp_boost}, "
                f"corrections={self._n_corrections})")