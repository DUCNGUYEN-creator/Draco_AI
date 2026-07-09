# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
thinking_engine.reasoning — the Cognition layer.

Reasoning is responsible ONLY for: generating hypotheses, searching for
solutions, planning, debating, and tool calling. It NEVER judges
correctness — that is the Verification layer's (reflection/hallucination)
exclusive job. Keeping this boundary crisp is the whole point of the
Infrastructure / Cognition / Verification split.
"""

from .thinking.tree_of_thought import TreeOfThoughts
from .thinking.self_consistency import SelfConsistency
from .thinking.chain_verifier import ChainOfThoughtVerifier
from .cognitive.abduction import AbductionEngine
from .cognitive.counterfactual import CounterfactualReasoner
from .cognitive.analogy import AnalogicalMapper
from .cognitive.hypothesis import HypothesisTester
from .debate.council import MultiAgentDebate
from .execution.controller import ReasoningController

__all__ = [
    "TreeOfThoughts",
    "SelfConsistency",
    "ChainOfThoughtVerifier",
    "AbductionEngine",
    "CounterfactualReasoner",
    "AnalogicalMapper",
    "HypothesisTester",
    "MultiAgentDebate",
    "ReasoningController",
]
