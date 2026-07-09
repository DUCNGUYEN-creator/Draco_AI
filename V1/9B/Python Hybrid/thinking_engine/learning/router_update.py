# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
RouterUpdater
==============
Updates the SelfEvolvingRouter's Beta-distribution parameters from a
feedback event — closing the loop so poor expert performance on a given
intent type gradually lowers that expert's posterior mean for that
intent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RouterUpdater:
    def __init__(self, evolving_router=None) -> None:
        self._router = evolving_router

    def update_from_feedback(self, feedback: Dict[str, Any], was_correct: bool) -> None:
        if self._router is None:
            return
        meta = feedback.get("metadata", {})
        intent_type = meta.get("intent_type", "chat")
        used_experts = meta.get("used_experts", [])
        for expert_id in used_experts:
            self._router.update(intent_type, expert_id, success=was_correct)
