# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
RecursiveReflectionLoop
==========================
Reasoning-side counterpart of reflection/self_reflection.py's
``recursive_critique``: iteratively deepens a *thought* (not yet a
final answer) by re-running ChainOfThoughtVerifier and, if unsound,
asking the caller-supplied refine function to patch it. Kept inside
reasoning/ because it operates purely on intermediate thoughts —
nothing here judges factual correctness against evidence, which stays
the Hallucination subsystem's job.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .chain_verifier import ChainOfThoughtVerifier


class RecursiveReflectionLoop:
    def __init__(self, verifier: Optional[ChainOfThoughtVerifier] = None, max_iter: int = 3) -> None:
        self.verifier = verifier or ChainOfThoughtVerifier()
        self.max_iter = max_iter

    def run(
        self,
        thoughts: List[str],
        refine_fn: Optional[Callable[[List[str], dict], List[str]]] = None,
    ) -> Tuple[List[str], dict]:
        current = list(thoughts)
        report = self.verifier.verify_thoughts(current)
        for _ in range(self.max_iter):
            if report["is_sound"] or refine_fn is None:
                break
            current = refine_fn(current, report)
            report = self.verifier.verify_thoughts(current)
        return current, report
