# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ConsensusChecker
==================
Simple keyword-overlap consensus check used to decide when a debate
round can stop early. Ported 1:1 from engine_v1.py's
``MultiAgentDebate._check_consensus`` (type hint fix: Iterable[str]).
"""

from __future__ import annotations

from typing import Iterable


class ConsensusChecker:
    def check(self, thoughts: Iterable[str], threshold: float = 0.75) -> bool:
        """If meaningful common tokens (len > 3) across ALL thoughts >= 6,
        consider consensus reached."""
        thought_list = list(thoughts)
        if len(thought_list) < 2:
            return True
        kw_sets = [set(t.lower().split()) for t in thought_list]
        base = kw_sets[0]
        common = base.intersection(*kw_sets[1:])
        meaningful = {w for w in common if len(w) > 3}
        return len(meaningful) >= 6
