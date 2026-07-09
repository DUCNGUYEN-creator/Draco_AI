# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
ChainOfThoughtVerifier
=========================
Reviews each thought step for logical consistency before the prompt is
compiled: contradiction, unsupported jump, circular reasoning, broken
causal chain, negation flip. Ported 1:1 from engine_v1.py's
``ChainOfThoughtVerifier``.

NOTE on architecture placement: this class lives in reasoning/thinking
(not reflection/hallucination) because it inspects the *shape* of the
model's own intermediate reasoning trace before any answer exists yet
— it is a reasoning-loop quality gate, not a post-hoc evidence
verifier. The Hallucination subsystem in reflection/hallucination/
consumes its output (cot_verification) as one signal among many but
does not duplicate this logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_CONTRADICTION_PAIRS = [
    (r"\btrue\b", r"\bfalse\b"),
    (r"\bđúng\b", r"\bsai\b"),
    (r"\btăng\b", r"\bgiảm\b"),
    (r"\bmore\b", r"\bless\b"),
]

_CAUSAL_MARKERS = [
    "vì", "bởi vì", "do đó", "vì vậy", "because", "therefore",
    "hence", "thus", "→", "⟹", "causes", "leads to",
]

_NEGATION_FLIP = [(r"\bkhông\s+(\w+)", r"\b\1\b"), (r"\bnot\s+(\w+)", r"\b\1\b")]


class ChainOfThoughtVerifier:
    def verify_thoughts(self, thoughts: List[str]) -> Dict[str, Any]:
        issues: List[str] = []
        score = 1.0
        combined = " ".join(thoughts).lower()

        for pat_a, pat_b in _CONTRADICTION_PAIRS:
            if re.search(pat_a, combined) and re.search(pat_b, combined):
                issues.append(f"Possible contradiction: '{pat_a}' vs '{pat_b}'")
                score -= 0.1

        for i, t in enumerate(thoughts):
            if len(t.split()) < 3:
                issues.append(f"Thought {i + 1} too brief — may be unsupported")
                score -= 0.05

        seen: set = set()
        for t in thoughts:
            key = frozenset(t.lower().split()[:6])
            if key in seen:
                issues.append("Circular reasoning detected in thoughts")
                score -= 0.15
                break
            seen.add(key)

        if len(thoughts) >= 2:
            reasoning_text = " ".join(thoughts[1:]).lower()
            if not any(m in reasoning_text for m in _CAUSAL_MARKERS):
                issues.append(
                    "No causal connector found — reasoning may lack explicit logic chain"
                )
                score -= 0.08

        for i in range(len(thoughts) - 1):
            a_lower = thoughts[i].lower()
            b_lower = thoughts[i + 1].lower()
            for neg_pat, pos_pat in _NEGATION_FLIP:
                for neg_match in re.finditer(neg_pat, a_lower):
                    term = neg_match.group(1)
                    if re.search(r"\b" + re.escape(term) + r"\b", b_lower):
                        issues.append(
                            f"Negation-flip: thought {i + 1} negates '{term}' "
                            f"but thought {i + 2} asserts it positively"
                        )
                        score -= 0.12
                        break

        score = max(0.0, min(1.0, score))
        return {
            "issues": issues,
            "score": score,
            "is_sound": score >= 0.7,
            "expert": "Debug Expert (EXPERT_CODE_2)",
        }

    def flag_thoughts(self, thoughts: List[str], verification: dict) -> List[str]:
        if verification["is_sound"]:
            return thoughts
        return [f"[?] {t}" if i < 2 else t for i, t in enumerate(thoughts)]
