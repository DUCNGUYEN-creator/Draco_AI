# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
IntentDetector
================
Hybrid weighted keyword scoring (TF-IDF-like) intent classifier. Ported
1:1 from engine_v1.py's ``IntentDetector`` — same keyword tables, same
normalized expert-boost mapping, same miro_tau formula. This keeps
expert routing numerically identical after the package split.
"""

from __future__ import annotations

import re
from typing import Dict, List

from ...constants import (
    EXPERT_CHAT,
    EXPERT_CODE,
    EXPERT_CODE_2,
    EXPERT_LANG_1,
    EXPERT_LANG_2,
    EXPERT_LANG_3,
    EXPERT_LANGUAGE,
    EXPERT_LOGIC,
    INTENT_CHAT,
    INTENT_CODE,
    INTENT_COMPARISON,
    INTENT_CREATIVE,
    INTENT_FACTUAL,
    INTENT_HOW_TO,
    INTENT_LOGIC,
    INTENT_MATH,
    INTENT_MEMORY,
    INTENT_WHY,
)
from ...utils.normalization import normalize_dict
from ...utils.text import detect_lang

PATTERNS: Dict[str, List] = {
    INTENT_MATH: [
        "tính", "bao nhiêu", "bằng", "cộng", "trừ", "nhân", "chia",
        ("=", 1.5), ("+", 1.2), ("-", 0.8), ("*", 1.2), ("/", 1.0),
        "phần trăm", ("sqrt", 2.0), ("log", 2.0), ("sin", 2.0), ("cos", 2.0),
    ],
    INTENT_LOGIC: [
        "nếu", "thì", ("logic", 2.0), ("suy luận", 2.0), ("chứng minh", 2.0),
        "vậy", ("mâu thuẫn", 2.0), ("tương đương", 2.0), ("prove", 2.0),
    ],
    INTENT_CODE: [
        ("code", 2.0), ("lập trình", 2.0), ("python", 2.0), ("javascript", 2.0),
        ("typescript", 2.0), ("function", 1.5), ("class", 1.5), ("bug", 2.0),
        ("error", 1.5), ("debug", 2.0), ("implement", 2.0),
        ("viết hàm", 2.0), ("def ", 3.0), ("import ", 2.0), ("```", 3.0),
    ],
    INTENT_CREATIVE: [
        ("viết truyện", 2.0), ("sáng tác", 2.0), ("thơ", 2.0), ("tưởng tượng", 1.5),
        ("kịch bản", 2.0), ("ý tưởng", 1.2), ("sáng tạo", 1.5),
        ("write story", 2.0), ("poem", 2.0),
    ],
    INTENT_FACTUAL: [
        ("là gì", 2.0), ("nghĩa là", 2.0), ("định nghĩa", 2.0),
        ("ai là", 1.5), "khi nào", "ở đâu", "năm nào",
        ("what is", 2.0), "when", "where", ("who", 1.5), ("define", 2.0),
    ],
    INTENT_HOW_TO: [
        ("làm sao", 2.0), ("cách", 1.5), ("như thế nào", 2.0),
        ("hướng dẫn", 2.0), ("các bước", 2.0), ("how to", 2.0), ("how do", 2.0),
    ],
    INTENT_WHY: [
        ("tại sao", 2.0), ("vì sao", 2.0), ("lý do", 1.5),
        ("nguyên nhân", 2.0), ("why", 2.0), ("reason", 1.5),
    ],
    INTENT_COMPARISON: [
        ("so sánh", 2.0), ("khác nhau", 2.0), ("giống nhau", 1.5),
        ("tốt hơn", 1.5), ("vs", 2.0), ("versus", 2.0),
        ("hay là", 1.0), ("compare", 2.0), ("difference", 2.0),
    ],
    INTENT_MEMORY: [
        ("nhớ rằng", 2.0), ("ghi nhớ", 2.0), ("lưu lại", 2.0),
        ("bạn có nhớ", 2.0), ("bạn biết", 1.5),
        ("remember", 2.0), ("forget", 2.0),
    ],
    INTENT_CHAT: [
        "xin chào", "hello", "cảm ơn", "bye", "hi", "ok", "oke",
        "thanks", "chào",
    ],
}


def _keyword_score(kws: List, tl: str) -> float:
    score = 0.0
    for entry in kws:
        if isinstance(entry, tuple):
            kw, w = entry
        else:
            kw, w = entry, 1.0
        if kw in tl:
            score += w
    return score


class IntentDetector:
    PATTERNS = PATTERNS

    def detect(self, text: str) -> dict:
        tl = text.lower()
        intent = INTENT_CHAT
        best = 0.0
        for itype, kws in self.PATTERNS.items():
            s = _keyword_score(kws, tl)
            if s > best:
                best = s
                intent = itype

        lang = detect_lang(text)
        entities = list(
            dict.fromkeys(
                re.findall(
                    r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẬẶỆỘỢỤ]"
                    r"[A-Za-zàáâãèéêìíòóôõùúăđơưạậặệộợụ0-9]+\b",
                    text,
                )
            )
        )[:5]
        pos = ["hay", "tốt", "tuyệt", "great", "love", "thích", "good", "awesome"]
        neg = [
            "tệ", "xấu", "dở", "ghét", "bad", "wrong", "horrible", "bực", "chán",
            "tức", "khó chịu", "frustrat", "annoying",
        ]
        sentiment = (
            "positive"
            if any(w in tl for w in pos)
            else "negative"
            if any(w in tl for w in neg)
            else "neutral"
        )
        creativity = (
            0.9
            if intent == INTENT_CREATIVE
            else 0.2
            if intent in (INTENT_MATH, INTENT_LOGIC, INTENT_CODE)
            else 0.6
        )
        if any(
            p in tl
            for p in ["bớt ảo", "thực tế hơn", "nghiêm túc", "chính xác", "factual", "bớt sáng tạo"]
        ):
            creativity = 0.1
        return {
            "intent": intent,
            "lang": lang,
            "entities": entities,
            "sentiment": sentiment,
            "creativity": creativity,
            "word_count": len(text.split()),
        }

    @staticmethod
    def _normalize_boost(raw: Dict[int, float]) -> Dict[int, float]:
        return normalize_dict(raw)

    def to_expert_boost(self, intent: dict) -> Dict[int, float]:
        i = intent["intent"]
        if i in (INTENT_MATH, INTENT_LOGIC):
            raw = {EXPERT_LOGIC: 0.4, EXPERT_CODE_2: 0.1}
        elif i == INTENT_CODE:
            raw = {EXPERT_CODE: 0.5, EXPERT_CODE_2: 0.2, EXPERT_LANGUAGE: 0.15}
        elif i == INTENT_CREATIVE:
            raw = {EXPERT_LANGUAGE: 0.4, EXPERT_LANG_1: 0.2}
        elif i in (INTENT_FACTUAL, INTENT_HOW_TO, INTENT_WHY, INTENT_COMPARISON):
            raw = {EXPERT_LANGUAGE: 0.25, EXPERT_LOGIC: 0.15, EXPERT_LANG_2: 0.1}
        else:
            raw = {EXPERT_CHAT: 0.35, EXPERT_LANG_3: 0.1}
        return self._normalize_boost(raw)

    def to_miro_tau(self, intent: dict) -> float:
        return 2.0 + intent["creativity"] * 6.0
