# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Beam Search
=============
New addition (marked NEW in the proposed architecture). Keeps the top-k
highest-scoring partial thought-sequences at each expansion step instead
of exploring exhaustively — a cheaper alternative to full MCTS when
``BudgetRouter`` allocates a tight compute budget.
"""

from __future__ import annotations

from typing import Callable, List, Tuple


def beam_search(
    branches: List[str],
    score_fn: Callable[[str], float],
    beam_width: int = 3,
    expand_fn: Callable[[str], List[str]] | None = None,
    max_depth: int = 3,
) -> List[Tuple[str, float]]:
    """Generic beam search over string "thoughts". If expand_fn is None,
    runs a single-step beam ranking (suitable for picking the top-k
    initial branches, e.g. before handing them to MCTSLight for deeper
    refinement). Returns [(thought, score), ...] sorted descending,
    length <= beam_width.
    """
    beam: List[Tuple[str, float]] = [(b, score_fn(b)) for b in branches]
    beam.sort(key=lambda x: x[1], reverse=True)
    beam = beam[:beam_width]

    if expand_fn is None:
        return beam

    for _ in range(max_depth - 1):
        candidates: List[Tuple[str, float]] = []
        for thought, _ in beam:
            for child in expand_fn(thought):
                candidates.append((child, score_fn(child)))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_width]

    return beam
