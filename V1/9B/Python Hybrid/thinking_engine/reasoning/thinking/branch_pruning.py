# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
BranchPruner
==============
New addition. Cheaply discards clearly-weak branches BEFORE they reach
expensive scoring (MCTS simulation, LLM scoring) — e.g. branches that
are near-duplicates of an already-kept branch, or trivially short.
Reduces wasted compute in TreeOfThoughts / GraphOfThoughts /
SelfConsistency when BudgetRouter has allocated a tight budget.
"""

from __future__ import annotations

from typing import List


class BranchPruner:
    def prune(
        self,
        branches: List[str],
        min_words: int = 2,
        dedup_similarity: float = 0.85,
    ) -> List[str]:
        kept: List[str] = []
        kept_word_sets = []
        for b in branches:
            words = set(b.lower().split())
            if len(words) < min_words:
                continue
            is_dup = False
            for kw in kept_word_sets:
                union = words | kw
                if not union:
                    continue
                jaccard = len(words & kw) / len(union)
                if jaccard >= dedup_similarity:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(b)
                kept_word_sets.append(words)
        return kept or branches  # never prune everything away
