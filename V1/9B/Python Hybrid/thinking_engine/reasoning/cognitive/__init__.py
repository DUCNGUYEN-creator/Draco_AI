# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.reasoning.cognitive — abduction/induction/deduction/analogy/causal reasoning."""

from .analogy import AnalogicalMapper
from .abduction import AbductionEngine
from .induction import InductiveReasoner
from .deduction import DeductiveReasoner
from .counterfactual import CounterfactualReasoner
from .hypothesis import HypothesisTester
from .causal_reasoning import CausalReasoner
from .decomposition import ProblemDecomposer
from .symbolic_reasoning import SymbolicReasoner

__all__ = [
    "AnalogicalMapper",
    "AbductionEngine",
    "InductiveReasoner",
    "DeductiveReasoner",
    "CounterfactualReasoner",
    "HypothesisTester",
    "CausalReasoner",
    "ProblemDecomposer",
    "SymbolicReasoner",
]
