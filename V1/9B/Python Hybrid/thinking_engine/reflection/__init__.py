# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection — the Verification layer.

reflection/ handles ONLY: self-critique, consistency checking,
confidence calibration, and answer rewriting. It NEVER generates new
reasoning (that's Cognition's job) and it depends on, but never
duplicates, reflection.hallucination — the dedicated, deep risk-
assessment subsystem. Hallucination produces a Report; Reflection
reads that Report and decides what to do about it.
"""

from .self_reflection import SelfReflection
from .consistency import ConsistencyChecker
from .confidence import ConfidenceScorer
from .confidence_calibrator import ConfidenceCalibrator
from .answer_rewriter import AnswerRewriter
from .critic import Critic
from . import hallucination

__all__ = [
    "SelfReflection",
    "ConsistencyChecker",
    "ConfidenceScorer",
    "ConfidenceCalibrator",
    "AnswerRewriter",
    "Critic",
    "hallucination",
]
