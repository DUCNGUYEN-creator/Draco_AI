# DracoAI V1 — thinking_engine/reflection/hallucination/factory/calibration_factory.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CalibrationFactory
=====================
Builds (and caches) calibration-method instances, resolving the
"ensemble" pseudo-name to calibration.ensemble.CalibrationEnsemble.
Unlike VerifierFactory/FusionFactory, calibration instances are
STATEFUL (they accumulate (raw_score, label) history) — the factory's
caching here is therefore load-bearing, not just a performance nicety:
callers MUST get the same instance back across calls to preserve
learned calibration state.

FIX (Bug #7): Added ``get_scoped(method, scope)`` public method so
``CalibrationPipeline._scoped()`` no longer needs to reach into the
private ``_instance_cache`` dict — the original code violated
encapsulation by doing ``self.factory._instance_cache[cache_key] = ...``
directly.
"""

from __future__ import annotations

from typing import Dict

from ..calibration.base import BaseCalibrator
from ..calibration.ensemble import CalibrationEnsemble
from ..registry.calibration_registry import CalibrationRegistry


class CalibrationFactory:
    def __init__(self, registry: CalibrationRegistry | None = None) -> None:
        self.registry = registry or CalibrationRegistry()
        self._instance_cache: Dict[str, object] = {}

    def get(self, name: str):
        """Get or create a calibrator instance by method name.
        Instances are cached so the same stateful calibrator is reused."""
        if name == "ensemble":
            return self._instance_cache.setdefault("ensemble", CalibrationEnsemble())
        if name not in self._instance_cache:
            self._instance_cache[name] = self.registry.create(name)
        return self._instance_cache[name]

    def get_scoped(self, method: str, scope: str):
        """Get or create a calibrator instance scoped by a namespace key
        (typically a verifier name or "_fused" for the post-fusion
        calibrator). This ensures each verifier gets its OWN fitted
        calibrator instance instead of sharing one model across verifiers
        with very different score distributions.

        This is the PUBLIC replacement for CalibrationPipeline's direct
        access to ``self._instance_cache``.

        Parameters
        ----------
        method : calibration method name ("platt", "isotonic", etc.)
        scope  : namespace key (e.g. "retrieval", "numerical", "_fused")
        """
        cache_key = f"{method}__{scope}"
        if cache_key not in self._instance_cache:
            if method == "ensemble":
                self._instance_cache[cache_key] = CalibrationEnsemble()
            else:
                self._instance_cache[cache_key] = self.registry.create(method)
        return self._instance_cache[cache_key]
