# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
PlanDecomposer
=================
Decomposes a complex question into ordered sub-goals using MCTS-scored
branches as a lightweight goal tree. Ported 1:1 from engine_v1.py's
``PlanDecomposer``.
"""

from __future__ import annotations

import re
from typing import Any, List

from ..constants import INTENT_CHAT, INTENT_CODE, INTENT_HOW_TO, INTENT_LOGIC, INTENT_MATH, INTENT_WHY
from ..reasoning.search.mcts import MCTSLight


class PlanDecomposer:
    def __init__(self, mcts: MCTSLight) -> None:
        self.mcts = mcts

    def decompose(
        self,
        question: str,
        intent: dict,
        max_subgoals: int = 4,
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
                    f"Question: {question}\n\n"
                    f"Create {max_subgoals} ordered steps to solve this, "
                    f"numbered 1 to {max_subgoals}. Be concrete and concise."
                )
                text = (
                    "<|im_start|>system\nYou are a step-by-step planning expert.\n"
                    f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=256)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    lines = [l.strip() for l in re.split(r"\n?\d+[.)]\s*", decoded) if l.strip()]
                    if len(lines) >= 2:
                        steps = lines[:max_subgoals]
                        best = self.mcts.search(question, steps)
                        return [f"[★] {t}" if t == best else t for t in steps]
            except Exception:
                pass

        if itype in (INTENT_MATH, INTENT_LOGIC):
            templates = [
                f"1. Identify known/unknown variables in: {question[:40]}",
                "2. Select applicable theorem or formula",
                "3. Apply step-by-step, checking units/constraints",
                "4. Verify result and edge cases",
            ]
        elif itype == INTENT_CODE:
            templates = [
                f"1. Clarify requirements & constraints for: {question[:40]}",
                "2. Design data structures and function signatures",
                "3. Implement core logic with error handling",
                "4. Write tests and document behavior",
            ]
        elif itype in (INTENT_HOW_TO, INTENT_WHY):
            templates = [
                f"1. Understand the context of: {question[:40]}",
                "2. Identify key factors or causes",
                "3. Explain mechanism or steps with examples",
                "4. Summarize with actionable conclusion",
            ]
        else:
            templates = [
                f"1. Parse the main topic of: {question[:40]}",
                "2. Retrieve relevant facts",
                "3. Structure and explain clearly",
                "4. Add context or caveats if needed",
            ]
        best = self.mcts.search(question, templates[:max_subgoals])
        return [f"[★] {t}" if t == best else t for t in templates[:max_subgoals]]
