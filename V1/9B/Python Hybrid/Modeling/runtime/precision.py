# DracoAI V1 — modeling/runtime/precision.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DynamicPrecisionManager — advisory dtype switching + VRAM budgeting.

Caller must act on current_dtype; this manager never mutates model weights.

NEW in this revision:
  ✅ FEAT-VRAM-BUDGET            : Hard-cap VRAM tracking.
     Records current estimated VRAM usage and enforces a soft ceiling.
     When usage exceeds the budget, recommend_upgrade() returns False,
     preventing unnecessary dtype escalation that would cause OOM.
  ✅ FEAT-PER-EXPERT-PRECISION   : Per-expert dtype advisory.
     Tracks per-expert logit health (overflow frequency) and returns
     a recommended dtype for each expert independently.  Experts
     handling complex logic (high overflow) get FP16; stable experts
     can stay ternary or INT4.
  ✅ FEAT-SELF-CORRECTION-SIGNAL : Expose confidence score.
     current_confidence() returns the EMA of max-prob token across
     recent steps.  health.py uses this to trigger self-correction
     loops when confidence drops below a threshold.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

__all__ = ["DynamicPrecisionManager"]

# Precision levels ordered from lowest to highest memory cost
_PREC_ORDER = ["ternary", "int4", "int8", "float16", "float32"]


class DynamicPrecisionManager:
    def __init__(
        self,
        overflow_thresh:    float     = 40.0,
        up_thresh:          float     = 0.05,
        down_thresh:        float     = 0.005,
        alpha:              float     = 0.1,
        initial_dtype:      np.dtype  = np.float16,
        vram_budget_gb:     float     = 0.0,   # 0 = no limit
        n_experts:          int       = 0,     # 0 = no per-expert tracking
        confidence_alpha:   float     = 0.05,  # EMA for confidence signal
    ):
        self._overflow_thresh  = overflow_thresh
        self._up_thresh        = up_thresh
        self._down_thresh      = down_thresh
        self._alpha            = alpha
        self._ema              = 0.0
        self._current_dtype    = np.dtype(initial_dtype)
        self._n_upgrades       = self._n_downgrades = self._n_steps = 0

        # VRAM budget tracking
        self._vram_budget_bytes: int  = int(vram_budget_gb * 1024 ** 3)
        self._vram_used_bytes:   int  = 0

        # Per-expert precision advisory
        self._n_experts = n_experts
        self._expert_overflow_ema: np.ndarray = (
            np.zeros(n_experts, dtype=np.float32) if n_experts > 0
            else np.empty(0, dtype=np.float32)
        )
        self._expert_dtypes: List[str] = (
            ["float16"] * n_experts if n_experts > 0 else []
        )

        # Confidence signal for self-correction
        self._confidence_alpha = confidence_alpha
        self._confidence_ema   = 1.0   # start optimistic

    # ── Core dtype management ─────────────────────────────────────────────────

    @property
    def current_dtype(self) -> np.dtype:
        return self._current_dtype

    def update(self, logits: np.ndarray) -> np.dtype:
        """Update overflow EMA and adjust global dtype advisory."""
        self._n_steps += 1
        if logits.size == 0:
            return self._current_dtype

        of = float((np.abs(logits) > self._overflow_thresh).mean())
        self._ema = self._alpha * of + (1 - self._alpha) * self._ema

        # Update confidence EMA (max softmax probability)
        logits_f = logits.astype(np.float64)
        probs = np.exp(logits_f - logits_f.max())
        probs /= probs.sum() + 1e-9
        max_prob = float(probs.max())
        self._confidence_ema = (
            self._confidence_alpha * max_prob
            + (1 - self._confidence_alpha) * self._confidence_ema
        )

        # Global dtype escalation / de-escalation
        if self.recommend_upgrade():
            if self._current_dtype == np.float16 and self._ema > self._up_thresh:
                self._current_dtype = np.dtype(np.float32)
                self._n_upgrades += 1
        if self._current_dtype == np.float32 and self._ema < self._down_thresh:
            self._current_dtype = np.dtype(np.float16)
            self._n_downgrades += 1

        return self._current_dtype

    # ── VRAM budget ───────────────────────────────────────────────────────────

    def register_vram(self, bytes_used: int) -> None:
        """Report current VRAM usage (called by TensorPool or allocators)."""
        self._vram_used_bytes = int(bytes_used)

    def recommend_upgrade(self) -> bool:
        """
        Return True if a dtype upgrade is allowed under the VRAM budget.

        When no budget is set (vram_budget_gb == 0) always returns True.
        When a budget is set, returns False when used > 90% of budget,
        preventing FP16→FP32 escalation that could cause OOM.
        """
        if self._vram_budget_bytes <= 0:
            return True
        return self._vram_used_bytes < 0.90 * self._vram_budget_bytes

    def vram_headroom_gb(self) -> float:
        """Remaining VRAM budget in GB (0 if no budget configured)."""
        if self._vram_budget_bytes <= 0:
            return float("inf")
        return max(0.0, (self._vram_budget_bytes - self._vram_used_bytes) / 1024**3)

    # ── Per-expert precision advisory ─────────────────────────────────────────

    def update_expert(self, expert_idx: int, logits_fragment: np.ndarray) -> str:
        """
        Update per-expert overflow EMA and return recommended dtype string.

        Returns one of: "ternary", "int4", "int8", "float16", "float32".

        Experts with consistently high overflow get float16; stable
        experts can run ternary or int4 to save memory/bandwidth.
        """
        if self._n_experts == 0 or expert_idx >= self._n_experts:
            return "float16"

        of = float((np.abs(logits_fragment) > self._overflow_thresh).mean())
        ema = self._expert_overflow_ema
        ema[expert_idx] = (
            self._alpha * of + (1 - self._alpha) * ema[expert_idx]
        )

        # Advisory: high overflow → keep float16, low → ternary safe
        v = float(ema[expert_idx])
        if v > self._up_thresh:
            rec = "float16"
        elif v > self._down_thresh:
            rec = "int8"
        else:
            rec = "ternary"

        self._expert_dtypes[expert_idx] = rec
        return rec

    def get_expert_dtype(self, expert_idx: int) -> str:
        """Return the last recommended dtype for a given expert."""
        if self._n_experts == 0 or expert_idx >= self._n_experts:
            return "float16"
        return self._expert_dtypes[expert_idx]

    def expert_summary(self) -> Dict[str, object]:
        """Summary of per-expert precision recommendations."""
        if self._n_experts == 0:
            return {}
        counts: Dict[str, int] = {}
        for d in self._expert_dtypes:
            counts[d] = counts.get(d, 0) + 1
        return {
            "n_experts":    self._n_experts,
            "dtype_counts": counts,
            "ema_overflow": [round(float(v), 5) for v in self._expert_overflow_ema],
        }

    # ── Self-correction signal ────────────────────────────────────────────────

    def current_confidence(self) -> float:
        """
        EMA of max-probability token over recent steps (0–1).

        Low values (< 0.1) indicate the model is uncertain — a signal
        for health.py to trigger a self-correction loop.
        """
        return float(self._confidence_ema)

    def needs_self_correction(self, threshold: float = 0.05) -> bool:
        """True when confidence EMA is below threshold."""
        return self._confidence_ema < threshold

    # ── Misc ──────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return dict(
            current_dtype    = str(self._current_dtype),
            overflow_ema     = round(self._ema, 5),
            n_upgrades       = self._n_upgrades,
            n_downgrades     = self._n_downgrades,
            steps            = self._n_steps,
            confidence_ema   = round(self._confidence_ema, 4),
            vram_used_gb     = round(self._vram_used_bytes / 1024**3, 3),
            vram_budget_gb   = round(self._vram_budget_bytes / 1024**3, 3),
            vram_headroom_gb = round(self.vram_headroom_gb(), 3),
        )

    def reset(self):
        self._ema = 0.0
        self._n_upgrades = self._n_downgrades = self._n_steps = 0
        self._confidence_ema = 1.0
        self._vram_used_bytes = 0
        if self._n_experts > 0:
            self._expert_overflow_ema[:] = 0.0
            self._expert_dtypes = ["float16"] * self._n_experts

    def __repr__(self) -> str:
        s = self.status()
        return (f"DynamicPrecisionManager(dtype={s['current_dtype']}, "
                f"ema={s['overflow_ema']:.4f}, conf={s['confidence_ema']:.3f}, "
                f"upgrades={s['n_upgrades']}, "
                f"vram_headroom={s['vram_headroom_gb']:.2f}GB)")