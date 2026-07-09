# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reflection.hallucination.verifiers
======================================================
Each verifier inspects ONE (claim, evidence, context) triple and
returns a VerificationResult. Verifiers NEVER mutate global state and
NEVER call the LLM to "fix" anything — they only judge. This is the
heart of the Hallucination subsystem, per the architecture's own
framing: "Thực ra verifier mới là trái tim."
"""

from .retrieval import RetrievalVerifier
from .reasoning import ReasoningVerifier
from .consistency import ConsistencyVerifier
from .contradiction import ContradictionVerifier
from .numerical import NumericalVerifier
from .symbolic import SymbolicVerifier
from .citation import CitationVerifier
from .planner import PlannerVerifier
from .tool import ToolVerifier
from .ensemble import VerifierEnsemble

__all__ = [
    "RetrievalVerifier",
    "ReasoningVerifier",
    "ConsistencyVerifier",
    "ContradictionVerifier",
    "NumericalVerifier",
    "SymbolicVerifier",
    "CitationVerifier",
    "PlannerVerifier",
    "ToolVerifier",
    "VerifierEnsemble",
]
