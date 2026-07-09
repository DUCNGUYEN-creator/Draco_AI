# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.reasoning.debate — Multi-Agent / Council debate."""

from .council import MultiAgentDebate
from .expert import ExpertProfile, EXPERT_NAMES, ROLE_TEMPLATES
from .arbitration import Arbitrator
from .voting import ConsensusChecker
from .consensus import build_consensus_report

__all__ = [
    "MultiAgentDebate",
    "ExpertProfile",
    "EXPERT_NAMES",
    "ROLE_TEMPLATES",
    "Arbitrator",
    "ConsensusChecker",
    "build_consensus_report",
]
