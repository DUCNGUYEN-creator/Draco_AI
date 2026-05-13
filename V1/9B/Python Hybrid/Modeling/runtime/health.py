# DracoAI V1 — modeling/runtime/health.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""HealthMonitor — NaN, saturation, expert collapse, memory checks."""
from __future__ import annotations
import logging
from typing import Callable, Optional
import numpy as np
from ..constants import HEALTH_SAT_THRESH, HEALTH_COLLAPSE_THRESH, HEALTH_MEM_WARN_MB

__all__ = ["HealthMonitor"]
logger = logging.getLogger(__name__)


class HealthMonitor:
    CRITICAL = "CRITICAL"
    WARNING  = "WARNING"

    def __init__(self, collapse_thresh: float = HEALTH_COLLAPSE_THRESH,
                 sat_thresh: float = HEALTH_SAT_THRESH,
                 mem_warn_mb: float = HEALTH_MEM_WARN_MB,
                 alert_cb: Optional[Callable] = None):
        self._collapse_thresh = collapse_thresh
        self._sat_thresh      = sat_thresh
        self._mem_warn_mb     = mem_warn_mb
        self._alert_cb        = alert_cb or (
            lambda lvl, msg: logger.warning("[HealthMonitor %s] %s", lvl, msg))
        self.reset()

    def reset(self):
        self._n_steps = self._n_nan = self._n_collapse = self._n_sat = self._n_mem_warn = 0

    def check_step(self, logits: np.ndarray, expert_counts: Optional[np.ndarray] = None):
        self._n_steps += 1
        if not np.all(np.isfinite(logits)):
            self._n_nan += 1
            self._alert_cb(self.CRITICAL, f"step {self._n_steps}: NaN/Inf in logits")
        if logits.size > 0 and float(np.abs(logits).max()) > self._sat_thresh:
            self._n_sat += 1
            self._alert_cb(self.WARNING,
                           f"step {self._n_steps}: logit saturation "
                           f"max={float(np.abs(logits).max()):.1f}")
        if expert_counts is not None and expert_counts.sum() > 0:
            fracs = expert_counts / expert_counts.sum()
            if float(fracs.max()) > self._collapse_thresh:
                top_e = int(fracs.argmax())
                self._n_collapse += 1
                self._alert_cb(self.WARNING,
                               f"step {self._n_steps}: expert collapse "
                               f"expert={top_e} frac={fracs[top_e]:.2%}")
        try:
            import resource as _res, sys as _sys
            rss = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss
            mb  = rss / 1024 if _sys.platform != "darwin" else rss / (1024 * 1024)
            if mb > self._mem_warn_mb:
                self._n_mem_warn += 1
                self._alert_cb(self.WARNING, f"step {self._n_steps}: high RSS={mb:.0f}MB")
        except Exception:
            pass

    def report(self) -> dict:
        return dict(steps_checked=self._n_steps, nan_events=self._n_nan,
                    collapse_events=self._n_collapse, saturation_events=self._n_sat,
                    mem_warn_events=self._n_mem_warn)

    def __repr__(self) -> str:
        r = self.report()
        return (f"HealthMonitor(steps={r['steps_checked']}, nan={r['nan_events']}, "
                f"collapse={r['collapse_events']}, sat={r['saturation_events']})")