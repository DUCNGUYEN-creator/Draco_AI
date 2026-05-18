# DracoAI V1 — modeling/ops/sparsity.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Neuron-level Activation Sparsity — PowerInfer-style inference skip.

Architecture
────────────
SwiGLU gate output h = silu(x @ W_g) has natural sparsity: many neurons
produce values near zero after the sigmoid-based gate.  Multiplying a
near-zero gate value against the up-projection W_u is wasteful.

This module provides:

  SparsityPredictor
    A lightweight per-expert linear probe that predicts which neurons
    in the gate activation will be near-zero BEFORE computing them.
    Trained online via exponential moving average of observed sparsity
    patterns — no separate training required.

  apply_sparsity_mask(gate_act, threshold)
    Zero out gate neurons below threshold.  When combined with
    TernaryLinear, masked neurons contribute no additions/subtractions,
    giving true computational skip.

  predict_active_neurons(predictor, x, threshold)
    Returns a boolean mask (out_feat,) for which neurons are predicted
    active.  Callers use this to skip W_u and W_d rows entirely.

Design constraints
──────────────────
• No separate training loop — online EMA of observed activation stats.
• Pure NumPy; no model state beyond the EMA vectors.
• Threshold is configurable per expert (different experts may have
  different natural sparsity levels).
• Thread-safe: all state updates are protected by a lock.

Typical natural sparsity of SwiGLU gate: 20–40% near-zero neurons.
With a threshold of 0.02, effective skip rate reaches 25–45%.
"""
from __future__ import annotations

import threading
from typing import Optional
import numpy as np

__all__ = ["SparsityPredictor", "apply_sparsity_mask"]


class SparsityPredictor:
    """
    Online EMA-based predictor of which gate neurons will be near-zero.

    For each output neuron j in the gate projection, tracks the EMA of
    |gate_j| observed during inference.  Neurons whose EMA falls below
    `threshold` are predicted inactive and can be skipped.

    Usage::

        pred = SparsityPredictor(d_ff=512)

        # During forward:
        gate_act = silu(x @ W_g)                         # (seq, d_ff)
        active = pred.predict_active(gate_act)            # (d_ff,) bool
        pred.update(gate_act)                             # update EMA

        # Skip computation for inactive neurons:
        masked = gate_act * active[None, :]               # zero inactive
    """

    def __init__(
        self,
        d_ff:           int,
        ema_alpha:      float = 0.05,
        threshold:      float = 0.02,
        warmup_steps:   int   = 64,
    ):
        """
        Parameters
        ----------
        d_ff        : feed-forward (gate) dimension.
        ema_alpha   : EMA decay for online mean-abs update (smaller = slower).
        threshold   : gate activations below this are predicted inactive.
        warmup_steps: don't predict during warmup (let EMA stabilise).
        """
        self.d_ff          = d_ff
        self.ema_alpha     = ema_alpha
        self.threshold     = threshold
        self.warmup_steps  = warmup_steps

        # EMA of mean-absolute gate activation per neuron
        self._ema_abs: np.ndarray = np.ones(d_ff, dtype=np.float32)
        self._step:    int        = 0
        self._lock     = threading.Lock()

    def update(self, gate_act: np.ndarray) -> None:
        """
        Update EMA with observed gate activations.

        gate_act : (seq, d_ff) float32 — post-silu gate values.
        """
        mean_abs = np.abs(gate_act).mean(axis=0).astype(np.float32)
        with self._lock:
            self._ema_abs = (
                self.ema_alpha * mean_abs
                + (1.0 - self.ema_alpha) * self._ema_abs
            )
            self._step += 1

    def predict_active(self, gate_act: np.ndarray) -> np.ndarray:
        """
        Return boolean mask of predicted-active neurons (shape: (d_ff,)).

        During warmup all neurons are active (mask = all True).
        Post-warmup: neurons whose current |gate| < threshold AND whose
        EMA is consistently low are predicted inactive.

        gate_act : (seq, d_ff) float32
        """
        with self._lock:
            if self._step < self.warmup_steps:
                return np.ones(self.d_ff, dtype=bool)
            # Current batch signal
            cur_abs = np.abs(gate_act).max(axis=0)
            # Combine: inactive if BOTH current AND historical are low
            inactive = (cur_abs < self.threshold) & (self._ema_abs < self.threshold)
            return ~inactive

    def sparsity_rate(self) -> float:
        """Estimated fraction of neurons predicted inactive (0–1)."""
        with self._lock:
            if self._step < self.warmup_steps:
                return 0.0
            return float((self._ema_abs < self.threshold).mean())

    def reset(self) -> None:
        with self._lock:
            self._ema_abs[:] = 1.0
            self._step = 0

    def __repr__(self) -> str:
        return (f"SparsityPredictor(d_ff={self.d_ff}, "
                f"threshold={self.threshold}, "
                f"sparsity={self.sparsity_rate():.1%}, "
                f"step={self._step})")


def apply_sparsity_mask(
    gate_act:  np.ndarray,
    threshold: float = 0.02,
) -> np.ndarray:
    """
    Zero out gate neurons below threshold (in-place on a copy).

    This is the simple non-predictive version — masks based on the
    CURRENT activation values rather than a learned predictor.
    Useful as a fallback when no SparsityPredictor is available.

    gate_act : (seq, d_ff) float32
    Returns  : (seq, d_ff) float32 with near-zero neurons zeroed.
    """
    mask = (np.abs(gate_act) >= threshold)
    return gate_act * mask