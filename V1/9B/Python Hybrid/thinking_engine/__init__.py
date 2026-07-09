# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine
=================
DracoAI's core reasoning engine, structured as:

    Engine = Infrastructure + Cognition + Verification

    Infrastructure  perception/, memory/, knowledge/, routing/, planning/, tools/
    Cognition       reasoning/
    Verification    reflection/ (including the deep reflection.hallucination
                    subsystem — a full Evidence -> Verification ->
                    Calibration -> Correlation -> Fusion -> Risk -> Report
                    pipeline)

This package is the structured, tested, GPL v3-licensed replacement
for the original monolithic ``engine_v1.py`` (~3,775 lines). Public
entry points:

    from thinking_engine import ThinkingEngine, EngineConfig
    from thinking_engine.reflection.hallucination import Assessor, HallucinationReport

See individual subpackage docstrings for details on each layer.
"""

from .config import EngineConfig, HallucinationConfig, MemoryConfig, ReasoningConfig, SafetyConfig
from .engine import ThinkingEngine, EngineComponents
from .state import ThinkingState
from .exceptions import ThinkingEngineError

__version__ = "1.0.0"

__all__ = [
    "ThinkingEngine",
    "EngineComponents",
    "EngineConfig",
    "HallucinationConfig",
    "MemoryConfig",
    "ReasoningConfig",
    "SafetyConfig",
    "ThinkingState",
    "ThinkingEngineError",
    "__version__",
]
