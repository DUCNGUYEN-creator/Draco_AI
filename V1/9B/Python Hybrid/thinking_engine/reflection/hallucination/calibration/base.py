# DracoAI V1 — thinking_engine/reflection/hallucination/calibration/base.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BaseCalibrator
=================
Abstract interface every calibration method implements. Concrete
calibrators are stateful (they accumulate (raw_score, label) samples)
but ALWAYS gracefully degrade to identity-mapping when too few samples
have been seen — never raise or return nonsense for an empty model.

Thread safety
-------------
``record()`` is called by ``Assessor.record_outcome()`` which the
Assessor docstring claims is "thread-safe". Without a lock, concurrent
``record()`` calls can corrupt ``_history`` (list.append is CPython-
atomic thanks to the GIL, but ``_fit()`` iterating over ``_history``
while another thread appends is NOT safe — iterator invalidation).
FIX (Bug #8): added a ``threading.Lock`` to serialise ``record()`` and
``calibrate()`` when the model is being re-fitted.
"""

from __future__ import annotations

import abc
import threading
from typing import List, Tuple

from ....exceptions import CalibrationError
from ..models.calibration import CalibrationModel, CalibrationPoint


class BaseCalibrator(abc.ABC):
    method_name: str = "base"
    min_samples_to_fit: int = 5

    def __init__(self) -> None:
        self._history: List[CalibrationPoint] = []
        self._fitted = False
        self._lock = threading.Lock()

    def record(self, raw_score: float, label: int) -> None:
        """Record a ground-truth observation. Thread-safe."""
        if label not in (0, 1):
            raise CalibrationError(f"label must be 0 or 1, got {label}")
        with self._lock:
            self._history.append(CalibrationPoint(raw_score=raw_score, label=label))
            if len(self._history) >= self.min_samples_to_fit:
                self._fit()
                self._fitted = True

    @property
    def n_samples(self) -> int:
        return len(self._history)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abc.abstractmethod
    def _fit(self) -> None:
        """Fit internal parameters using self._history. Implementations
        should be cheap enough to re-run on every record() call (online
        learning), not a one-shot batch fit."""

    @abc.abstractmethod
    def _predict(self, raw_score: float) -> float:
        """Map a raw score to a calibrated probability, ASSUMING the
        model is fitted. Never called directly by external code —
        use calibrate() instead, which handles the cold-start case."""

    def calibrate(self, raw_score: float) -> float:
        """Calibrate a raw score. Thread-safe (reads fitted state under lock)."""
        with self._lock:
            if not self._fitted:
                return max(0.0, min(1.0, raw_score))  # identity fallback, cold start
            return max(0.0, min(1.0, self._predict(raw_score)))

    def export_model(self) -> CalibrationModel:
        with self._lock:
            return CalibrationModel(
                method=self.method_name,
                params=self._export_params(),
                n_samples=self.n_samples,
                fitted=self._fitted,
                history=[(p.raw_score, p.label) for p in self._history],
            )

    def _export_params(self) -> dict:
        return {}
