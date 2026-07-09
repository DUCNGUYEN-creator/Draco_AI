# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Arbitrator
============
Expert 0 (Logic/Math Expert) acts as final arbiter of a council debate,
picking the thought with highest keyword overlap with the question.
Ported 1:1 from engine_v1.py's ``MultiAgentDebate._arbitrate`` (V2:
no longer receives debate_log).
"""

from __future__ import annotations

from typing import Dict

from .expert import EXPERT_NAMES


class Arbitrator:
    def arbitrate(
        self,
        final_thoughts: Dict[int, str],
        rounds_done: int,
        question: str,
        intent: dict,
    ) -> str:
        q_words = set(question.lower().split())
        best_id = max(
            final_thoughts,
            key=lambda eid: len(q_words & set(final_thoughts[eid].lower().split())),
        )
        best_name = EXPERT_NAMES.get(best_id, f"Expert{best_id}")
        return (
            f"[COUNCIL ARBITRATION — {rounds_done} round(s), {len(final_thoughts)} experts] "
            f"Arbiter (Logic/Math Expert) selects: {best_name}'s approach. "
            f"Rationale: highest alignment with question semantics. "
            f"Final stance: {final_thoughts[best_id][:200]}"
        )
