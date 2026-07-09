# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
CounterfactualReasoner
=========================
Generates a "what-if" branch for logic/legal/policy questions. Ported
1:1 from engine_v1.py's ``CounterfactualReasoner`` (V2 broader
trigger patterns: "nếu/giả sử/what if").
"""

from __future__ import annotations

import re
from typing import Any

from ...constants import INTENT_LOGIC, INTENT_WHY

_TRIGGER_INTENTS = {INTENT_LOGIC, INTENT_WHY}
_CF_PATTERNS = [
    r"nếu\s+không", r"what\s+if\s+not", r"giả sử", r"suppose",
    r"hypothetically", r"nếu\s+\w+\s+không",
    r"\bnếu\b", r"\bwhat\s+if\b",
]


class CounterfactualReasoner:
    def is_applicable(self, query: str, intent: dict) -> bool:
        itype = intent.get("intent", "chat")
        ql = query.lower()
        has_cf_pattern = any(re.search(p, ql) for p in _CF_PATTERNS)
        return has_cf_pattern or itype in _TRIGGER_INTENTS

    def generate(
        self,
        question: str,
        intent: dict,
        bridge: Any = None,
        tokenizer: Any = None,
    ) -> str:
        entities = intent.get("entities", [])
        subj = entities[0] if entities else "the subject"

        if (
            bridge is not None
            and hasattr(bridge, "is_connected")
            and bridge.is_connected()
            and tokenizer is not None
        ):
            try:
                prompt = (
                    f"Question: {question}\n\n"
                    f"Now reason counterfactually: what if '{subj}' were NOT the case? "
                    f"Describe how the outcome or reasoning would change."
                )
                text = (
                    "<|im_start|>system\nYou are a counterfactual reasoning expert.\n"
                    f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                ids = tokenizer.encode(text, add_bos=True)
                out = bridge.generate(ids, max_new_tokens=180)
                if out:
                    decoded = tokenizer.decode(out).strip()
                    if decoded:
                        return f"[COUNTERFACTUAL] {decoded}"
            except Exception:
                pass

        return (
            f"[COUNTERFACTUAL] If '{subj}' were NOT the case: "
            f"the reasoning chain would diverge at the first premise. "
            f"Alternative outcome: the conclusion would likely be negated or weakened. "
            f"Consistency check: the factual answer should hold against this counterfactual."
        )
