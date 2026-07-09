# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Small text utilities shared by perception/language modules."""

from __future__ import annotations

import re
from typing import List

_VIET_CHARS = set(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
)


def detect_lang(text: str) -> str:
    tl = text.lower()
    return "vi" if any(c in _VIET_CHARS for c in tl) else "en"


def tokenize_words(text: str) -> List[str]:
    return text.split()


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def extract_capitalized_entities(text: str, limit: int = 5) -> List[str]:
    pattern = (
        r"\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐƠƯẠẬẶỆỘỢỤ]"
        r"[A-Za-zàáâãèéêìíòóôõùúăđơưạậặệộợụ0-9]+\b"
    )
    seen: List[str] = []
    for m in re.findall(pattern, text):
        if m not in seen:
            seen.append(m)
        if len(seen) >= limit:
            break
    return seen
