# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ExpertSelector
================
Picks the top-N experts for a debate/council round given an
expert_boost dict, clamped to [1, 8] experts — formalizes the
``max_experts`` clamping logic that engine_v1.py applied ad-hoc inside
``run_full_council``.
"""

from __future__ import annotations

from typing import Dict, List

from ..constants import N_EXPERTS


class ExpertSelector:
    def select(self, expert_boost: Dict[int, float], max_experts: int = 4) -> List[int]:
        max_experts = max(1, min(max_experts, N_EXPERTS))
        ranked = sorted(expert_boost.items(), key=lambda kv: kv[1], reverse=True)
        selected = [eid for eid, _ in ranked[:max_experts]]
        # Fill remaining slots with unused expert ids if expert_boost was sparse
        if len(selected) < max_experts:
            for eid in range(N_EXPERTS):
                if eid not in selected:
                    selected.append(eid)
                if len(selected) >= max_experts:
                    break
        return selected[:max_experts]
