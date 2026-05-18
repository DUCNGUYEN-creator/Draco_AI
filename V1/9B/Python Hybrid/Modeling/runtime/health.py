# DracoAI V1 — modeling/runtime/health.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
HealthMonitor — NaN, saturation, expert collapse, memory, self-correction.

NEW in this revision:
  ✅ FEAT-SELF-CORRECTION-SIGNAL : check_step() returns a SelfCorrectionSignal
     named-tuple indicating whether the model should re-run the current
     block.  Triggered when the top-token confidence is below
     `correction_conf_thresh` for `correction_patience` consecutive steps.
     The caller (transformer.py generate()) may use this to:
       • Temporarily increase add_noise (diversify routing)
       • Re-sample from a wider distribution (higher temperature)
     Signal is a recommendation only — never mutates model state.
  ✅ FEAT-ADVERSARIAL-EXPERT-GUARD: detect_adversarial_expert() checks
     whether any single expert dominates with abnormally high router
     logits, which is a Trojan Horse indicator.  Returns the offending
     expert index or -1 if clean.  Flagged experts are tracked so the
     caller can route around them.
  ✅ FEAT-LOGIT-ENTROPY-TRACK    : tracks entropy EMA of the output
     distribution.  Abnormally low entropy (model is over-certain)
     combined with high saturation may indicate hallucination mode.
"""
from __future__ import annotations
import logging
from collections import namedtuple
from typing import Callable, List, Optional
import numpy as np

from ..constants import HEALTH_SAT_THRESH, HEALTH_COLLAPSE_THRESH, HEALTH_MEM_WARN_MB

__all__ = ["HealthMonitor", "SelfCorrectionSignal"]

logger = logging.getLogger(__name__)

SelfCorrectionSignal = namedtuple(
    "SelfCorrectionSignal",
    ["should_correct", "confidence", "consecutive_low", "reason"],
)


class HealthMonitor:
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"

    def __init__(
        self,
        collapse_thresh:        float    = HEALTH_COLLAPSE_THRESH,
        sat_thresh:             float    = HEALTH_SAT_THRESH,
        mem_warn_mb:            float    = HEALTH_MEM_WARN_MB,
        alert_cb:               Optional[Callable] = None,
        # Self-correction parameters
        correction_conf_thresh: float    = 0.05,
        correction_patience:    int      = 3,
        # Adversarial expert guard
        adversarial_logit_thresh: float  = 20.0,
        adversarial_dominance:    float  = 0.95,
        # Entropy tracking
        entropy_alpha:          float    = 0.05,
    ):
        """
        Parameters
        ----------
        correction_conf_thresh : Max-prob below this triggers self-correction.
        correction_patience    : Consecutive low-confidence steps before signal.
        adversarial_logit_thresh : Router logit magnitude threshold for Trojan detection.
        adversarial_dominance  : Fraction of tokens to single expert for Trojan detection.
        entropy_alpha          : EMA decay for output distribution entropy tracking.
        """
        self._collapse_thresh          = collapse_thresh
        self._sat_thresh               = sat_thresh
        self._mem_warn_mb              = mem_warn_mb
        self._alert_cb                 = alert_cb or (
            lambda lvl, msg: logger.warning("[HealthMonitor %s] %s", lvl, msg))

        # Self-correction state
        self._correction_conf_thresh   = correction_conf_thresh
        self._correction_patience      = correction_patience
        self._consecutive_low_conf     = 0

        # Adversarial guard state
        self._adversarial_logit_thresh = adversarial_logit_thresh
        self._adversarial_dominance    = adversarial_dominance
        self._flagged_experts: List[int] = []

        # Entropy EMA
        self._entropy_alpha = entropy_alpha
        self._entropy_ema   = 0.0

        self.reset()

    def reset(self):
        self._n_steps = self._n_nan = self._n_collapse = self._n_sat = self._n_mem_warn = 0
        self._n_self_corrections = 0
        self._n_adversarial      = 0
        self._consecutive_low_conf = 0
        self._entropy_ema          = 0.0
        self._flagged_experts.clear()

    # ── Primary check ─────────────────────────────────────────────────────────

    def check_step(
        self,
        logits:        np.ndarray,
        expert_counts: Optional[np.ndarray] = None,
        router_logits: Optional[np.ndarray] = None,
    ) -> SelfCorrectionSignal:
        """
        Run all health checks for one decode step.

        Parameters
        ----------
        logits        : Final token logits (vocab_size,) float64.
        expert_counts : Per-expert token counts for collapse detection.
        router_logits : Raw router logits for adversarial detection.

        Returns
        -------
        SelfCorrectionSignal — recommendation to re-run the step.
        """
        self._n_steps += 1

        # ── NaN / Inf ─────────────────────────────────────────────────
        if not np.all(np.isfinite(logits)):
            self._n_nan += 1
            self._alert_cb(self.CRITICAL, f"step {self._n_steps}: NaN/Inf in logits")

        # ── Saturation ────────────────────────────────────────────────
        if logits.size > 0 and float(np.abs(logits).max()) > self._sat_thresh:
            self._n_sat += 1
            self._alert_cb(self.WARNING,
                           f"step {self._n_steps}: logit saturation "
                           f"max={float(np.abs(logits).max()):.1f}")

        # ── Expert collapse ───────────────────────────────────────────
        if expert_counts is not None and expert_counts.sum() > 0:
            fracs = expert_counts / expert_counts.sum()
            if float(fracs.max()) > self._collapse_thresh:
                top_e = int(fracs.argmax())
                self._n_collapse += 1
                self._alert_cb(self.WARNING,
                               f"step {self._n_steps}: expert collapse "
                               f"expert={top_e} frac={fracs[top_e]:.2%}")

        # ── Entropy EMA ───────────────────────────────────────────────
        if logits.size > 0:
            lf = logits.astype(np.float64)
            probs = np.exp(lf - lf.max())
            probs /= probs.sum() + 1e-9
            entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
            self._entropy_ema = (
                self._entropy_alpha * entropy
                + (1 - self._entropy_alpha) * self._entropy_ema
            )
            confidence = float(probs.max())
        else:
            confidence = 1.0

        # ── Self-correction signal ────────────────────────────────────
        if confidence < self._correction_conf_thresh:
            self._consecutive_low_conf += 1
        else:
            self._consecutive_low_conf = 0

        should_correct = self._consecutive_low_conf >= self._correction_patience
        if should_correct:
            self._n_self_corrections += 1
            reason = (f"conf={confidence:.3f} < thresh={self._correction_conf_thresh} "
                      f"for {self._consecutive_low_conf} steps")
        else:
            reason = ""

        # ── Adversarial expert check ──────────────────────────────────
        if router_logits is not None:
            adv = self.detect_adversarial_expert(router_logits)
            if adv >= 0 and adv not in self._flagged_experts:
                self._flagged_experts.append(adv)
                self._n_adversarial += 1
                self._alert_cb(self.WARNING,
                               f"step {self._n_steps}: adversarial expert "
                               f"candidate idx={adv}")

        # ── Memory ────────────────────────────────────────────────────
        try:
            import resource as _res, sys as _sys
            rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
            mb  = rss / 1024 if _sys.platform != "darwin" else rss / (1024 * 1024)
            if mb > self._mem_warn_mb:
                self._n_mem_warn += 1
                self._alert_cb(self.WARNING,
                               f"step {self._n_steps}: high RSS={mb:.0f}MB")
        except Exception:
            pass

        return SelfCorrectionSignal(
            should_correct   = should_correct,
            confidence       = confidence,
            consecutive_low  = self._consecutive_low_conf,
            reason           = reason,
        )

    # ── Adversarial detection ─────────────────────────────────────────────────

    def detect_adversarial_expert(self, router_logits: np.ndarray) -> int:
        """
        Detect potential Trojan expert: one expert with abnormally high
        logits AND taking the majority of tokens.

        router_logits : (seq, n_experts) float32

        Returns expert index (0-based) if suspicious, else -1.
        """
        if router_logits.ndim != 2 or router_logits.shape[1] < 2:
            return -1

        # Check max logit magnitude
        max_logit = float(np.abs(router_logits).max())
        if max_logit < self._adversarial_logit_thresh:
            return -1

        # Find which expert has the max logit in each row
        top_experts = router_logits.argmax(axis=-1)    # (seq,)
        n_experts   = router_logits.shape[1]

        for e in range(n_experts):
            dominance = float((top_experts == e).mean())
            if dominance > self._adversarial_dominance:
                return int(e)
        return -1

    @property
    def flagged_experts(self) -> List[int]:
        """List of expert indices flagged as potentially adversarial."""
        return list(self._flagged_experts)

    def clear_flagged_experts(self) -> None:
        self._flagged_experts.clear()

    # ── Reporting ─────────────────────────────────────────────────────────────

    def report(self) -> dict:
        return dict(
            steps_checked        = self._n_steps,
            nan_events           = self._n_nan,
            collapse_events      = self._n_collapse,
            saturation_events    = self._n_sat,
            mem_warn_events      = self._n_mem_warn,
            self_correction_events = self._n_self_corrections,
            adversarial_events   = self._n_adversarial,
            flagged_experts      = list(self._flagged_experts),
            entropy_ema          = round(self._entropy_ema, 4),
        )

    def __repr__(self) -> str:
        r = self.report()
        return (f"HealthMonitor(steps={r['steps_checked']}, nan={r['nan_events']}, "
                f"collapse={r['collapse_events']}, sat={r['saturation_events']}, "
                f"corrections={r['self_correction_events']}, "
                f"adversarial={r['adversarial_events']})")