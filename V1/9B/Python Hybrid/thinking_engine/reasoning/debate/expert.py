# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Expert profile table
=======================
Static expert-name and role-template tables shared by council.py,
arbitration.py and voting.py. Ported 1:1 from the dicts embedded in
engine_v1.py's ``MultiAgentDebate``, split out so other modules can
look up a role hint without importing the full debate orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ...constants import (
    EXPERT_CODE_0,
    EXPERT_CODE_1,
    EXPERT_CODE_2,
    EXPERT_CODE_3,
    EXPERT_LANG_0,
    EXPERT_LANG_1,
    EXPERT_LANG_2,
    EXPERT_LANG_3,
)

EXPERT_NAMES: Dict[int, str] = {
    EXPERT_CODE_0: "Logic/Math Expert",
    EXPERT_CODE_1: "Code Expert",
    EXPERT_CODE_2: "Debug Expert",
    EXPERT_CODE_3: "System Expert",
    EXPERT_LANG_0: "Language Expert",
    EXPERT_LANG_1: "Chat Expert",
    EXPERT_LANG_2: "Creative Expert",
    EXPERT_LANG_3: "Memory Expert",
}

ROLE_TEMPLATES: Dict[int, str] = {
    EXPERT_CODE_0: "Apply formal step-by-step logic/math rules. Prioritize correctness.",
    EXPERT_CODE_1: "Write clean, efficient, well-structured code or pseudocode.",
    EXPERT_CODE_2: "Check for edge cases, bugs, and logical inconsistencies.",
    EXPERT_CODE_3: "Consider system-level concerns: memory, performance, security.",
    EXPERT_LANG_0: "Provide structured, informative context with clear explanation.",
    EXPERT_LANG_1: "Keep response natural, conversational, and user-friendly.",
    EXPERT_LANG_2: "Explore creative angles, analogies, and novel framings.",
    EXPERT_LANG_3: "Cross-reference prior knowledge/memory for factual grounding.",
}


@dataclass
class ExpertProfile:
    expert_id: int
    name: str
    role_hint: str

    @classmethod
    def lookup(cls, expert_id: int) -> "ExpertProfile":
        return cls(
            expert_id=expert_id,
            name=EXPERT_NAMES.get(expert_id, f"Expert{expert_id}"),
            role_hint=ROLE_TEMPLATES.get(expert_id, "Provide a balanced response."),
        )
