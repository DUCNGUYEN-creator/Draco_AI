# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
AbductionEngine
=================
For "why / vì sao" questions with no clear stated cause: enumerates
candidate hypotheses and ranks them via MCTS + KG priors. Ported 1:1
from engine_v1.py's ``AbductionEngine``.
"""

from __future__ import annotations

import re
from typing import Any, List, TYPE_CHECKING

from ..search.mcts import MCTSLight

if TYPE_CHECKING:  # pragma: no cover
    from ...knowledge.knowledge_graph import KnowledgeGraph

_TRIGGER_WORDS = ["vì sao", "tại sao", "why", "what caused", "lý do"]


class AbductionEngine:
    def __init__(self, mcts: MCTSLight) -> None:
        self.mcts = mcts

    def is_applicable(self, query: str, intent: dict) -> bool:
        ql = query.lower()
        return any(w in ql for w in _TRIGGER_WORDS)

    def generate_hypotheses(
        self,
        query: str,
        kg: "KnowledgeGraph",
        intent: dict,
        n: int = 4,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> List[str]:
        entities = intent.get("entities", [])

        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                prompt = (
                    f"Question: {query}\n\n"
                    f"List exactly {n} distinct hypotheses (possible explanations) "
                    f"numbered 1 to {n}. Be concise."
                )
                text = (
                    "<|im_start|>system\nYou are an abductive reasoning expert.\n"
                    f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=256)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    lines = [l.strip() for l in re.split(r"\n?\d+[.)]\s*", decoded) if l.strip()]
                    if len(lines) >= 2:
                        best = self.mcts.search(query, lines[:n])
                        return [f"[BEST] {h}" if h == best else h for h in lines[:n]]
            except Exception:
                pass

        templates: List[str] = [
            f"Hypothesis {i + 1}: related to {entities[i % len(entities)] if entities else 'unknown factor'}"
            for i in range(n)
        ]
        best = self.mcts.search(query, templates)
        return [f"[BEST] {t}" if t == best else t for t in templates]
