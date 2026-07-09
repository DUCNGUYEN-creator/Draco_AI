# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PlanOptimizer
================
New addition. Post-processes a flat ordered-step plan (from
GoalDecomposer/PlanDecomposer) to merge near-duplicate steps and drop
steps that add no new keywords over the previous step — the planning
analogue of reasoning/thinking/branch_pruning.py.
"""

from __future__ import annotations

from typing import List


class PlanOptimizer:
    def optimize(self, steps: List[str], min_new_word_ratio: float = 0.2) -> List[str]:
        if not steps:
            return steps
        optimized = [steps[0]]
        seen_words = set(steps[0].lower().split())
        for step in steps[1:]:
            words = set(step.lower().split())
            new_words = words - seen_words
            ratio = len(new_words) / max(len(words), 1)
            if ratio >= min_new_word_ratio:
                optimized.append(step)
                seen_words |= words
        return optimized or steps
