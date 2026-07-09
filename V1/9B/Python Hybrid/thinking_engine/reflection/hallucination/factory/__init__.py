# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.factory
====================================================
Factories sit on top of registry/* and add construction-time policy:
caching constructed instances (verifiers/fusion/calibration objects are
stateless-ish and safe to reuse), and resolving "ensemble" pseudo-names
to the right composite class. assessor.py and strategy/*.py depend on
factory/*, never on registry/* directly — this is what lets the
registry stay a pure name→constructor map while the factory owns
caching/composition concerns.
"""

from .verifier_factory import VerifierFactory
from .fusion_factory import FusionFactory
from .calibration_factory import CalibrationFactory

__all__ = ["VerifierFactory", "FusionFactory", "CalibrationFactory"]
