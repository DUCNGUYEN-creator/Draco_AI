# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SelfReflection
=================
Self-critique pass over a generated answer: flags absolutist hallucination
patterns ("100%", "luôn luôn đúng"), missing expected facts, and word
repetition. Ported 1:1 from engine_v1.py's ``SelfReflection``.

NOTE on architecture placement: this is a *style/coherence* critique —
it never consults retrieved evidence or runs a verifier ensemble. Deep,
evidence-grounded risk assessment is reflection.hallucination's job;
SelfReflection only decides "does this read like the kind of text that
tends to be wrong" and whether a refine pass is worth attempting.
"""

from __future__ import annotations

import re
from typing import List

_HALL_PATTERNS = [r"\b100%\b", r"chắc chắn hoàn toàn", r"luôn luôn đúng", r"never wrong"]


class SelfReflection:
    HALL_PATTERNS = _HALL_PATTERNS

    def critique(self, answer: str, question: str, facts: List[dict]) -> dict:
        issues: List[str] = []
        score = 1.0
        if not answer or len(answer.strip()) < 5:
            issues.append("Câu trả lời quá ngắn hoặc rỗng")
            score -= 0.3
        else:
            for pat in self.HALL_PATTERNS:
                if re.search(pat, answer, re.IGNORECASE):
                    issues.append(f"Tuyệt đối hóa: {pat}")
                    score -= 0.1
            for f in facts[:5]:
                k = f.get("key", "")
                v = str(f.get("value", ""))
                if k.lower() in question.lower() and v.lower() not in answer.lower():
                    issues.append(f"Có thể thiếu: {k}={v}")
                    score -= 0.05
            words = answer.split()
            if len(words) > 10 and len(set(words)) / len(words) < 0.4:
                issues.append("Lặp từ nhiều")
                score -= 0.15
        score = max(0.0, min(1.0, score))
        return {"issues": issues, "score": score, "should_refine": score < 0.7}

    def build_refine_prompt(self, orig: str, critique: dict) -> str:
        issues = "\n".join(f"- {i}" for i in critique["issues"]) or "---"
        return (
            f"[CRITIQUE]\n{issues}\n\n"
            f"[ORIGINAL]\n{orig}\n\n"
            f"[TASK] Cải thiện câu trả lời, sửa các vấn đề trên.\n"
            f"[REFINED ANSWER]"
        )
