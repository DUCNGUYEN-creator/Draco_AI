# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
GoalDecomposer
=================
For complex multi-step planning goals ("Lên kế hoạch học Python 30
ngày"), uses deeper MCTS (rollout_depth=20) to produce a goal tree.
Ported 1:1 from engine_v1.py's ``GoalDecomposer``.
"""

from __future__ import annotations

import re
from typing import Any, List

from ..constants import INTENT_CHAT
from ..reasoning.search.mcts import MCTSLight


class GoalDecomposer:
    def __init__(self) -> None:
        self.mcts = MCTSLight(n_sim=15, max_rollout_depth=20)

    def decompose(
        self,
        question: str,
        intent: dict,
        max_subgoals: int = 6,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> List[str]:
        itype = intent.get("intent", INTENT_CHAT)

        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                prompt = (
                    f"Goal: {question}\n\n"
                    f"Break this down into exactly {max_subgoals} concrete sub-goals, "
                    f"numbered 1 to {max_subgoals}. Each sub-goal on its own line."
                )
                text = (
                    "<|im_start|>system\nYou are a goal-decomposition expert.\n"
                    f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=300)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    lines = [l.strip() for l in re.split(r"\n?\d+[.)]\s*", decoded) if l.strip()]
                    if len(lines) >= 2:
                        templates = lines[:max_subgoals]
                        best = self.mcts.search(question, templates)
                        return [f"[★ MAIN GOAL] {t}" if t == best else t for t in templates]
            except Exception:
                pass

        templates = self._templates(question, itype, max_subgoals)
        best = self.mcts.search(question, templates)
        return [f"[★ MAIN GOAL] {t}" if t == best else t for t in templates]

    def _templates(self, question: str, itype: str, n: int) -> List[str]:
        base = [
            f"Understand and clarify the goal: '{question[:40]}'",
            "Break down into weekly/daily milestones",
            "Identify prerequisites and learning resources",
            "Set measurable success criteria",
            "Plan review and adjustment checkpoints",
            "Define final deliverable or outcome",
        ]
        return base[:n]
