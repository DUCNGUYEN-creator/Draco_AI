# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.registry
=====================================================
Plugin registries for verifiers, fusion strategies, and calibration
methods. New implementations register themselves here once; assessor.py
and strategy/* never need code changes to pick them up. This is the
mechanism that fulfils the architecture's design goal: "Sau này plugin
mới chỉ cần register. Không sửa code cũ."
"""

from .verifier_registry import VerifierRegistry
from .fusion_registry import FusionRegistry
from .calibration_registry import CalibrationRegistry

__all__ = ["VerifierRegistry", "FusionRegistry", "CalibrationRegistry"]
