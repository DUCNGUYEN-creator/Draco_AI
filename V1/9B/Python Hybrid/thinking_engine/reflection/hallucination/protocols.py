# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
hallucination.protocols
========================
Structural Protocol types that third-party verifiers, fusion methods,
and calibration implementations must satisfy to be registerable via
registry/*. These mirror the concrete base classes but as
runtime_checkable Protocols — so a plugin implemented in a completely
separate package (no dependency on thinking_engine's concrete ABCs) can
still be type-checked via isinstance(obj, VerifierProtocol).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class VerifierProtocol(Protocol):
    name: str

    def verify(self, claim: str, evidence: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


@runtime_checkable
class FusionProtocol(Protocol):
    method_name: str

    def fuse(self, signals: List) -> Any:
        ...


@runtime_checkable
class CalibratorProtocol(Protocol):
    method_name: str

    def record(self, raw_score: float, label: int) -> None:
        ...

    def calibrate(self, raw_score: float) -> float:
        ...
