# DracoAI V1 — tokenizer/pretokenizer/languages/thai.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""Thai word segmentation with library fallback to character-level split."""

from typing import List

_THAI_LIB = "none"

try:
    from pythainlp.tokenize import word_tokenize as _pythainlp_tok  # type: ignore
    _THAI_LIB = "pythainlp"
except ImportError:
    pass

if _THAI_LIB == "none":
    try:
        import thai_segmenter  # type: ignore
        _THAI_LIB = "thai_segmenter"
    except ImportError:
        pass


def split_thai(text: str) -> List[str]:
    """
    Segment Thai text into word-like units.

    Uses pythainlp (preferred) or thai_segmenter if installed.
    Falls back to character-level splitting (each codepoint its own
    segment) when no Thai NLP library is available.
    """
    if not text:
        return []

    if _THAI_LIB == "pythainlp":
        try:
            words = _pythainlp_tok(text, engine="newmm", keep_whitespace=True)
            return [w for w in words if w]
        except Exception:
            pass

    if _THAI_LIB == "thai_segmenter":
        try:
            import thai_segmenter as _ts  # type: ignore
            return [w for w in _ts.tokenize(text) if w]
        except Exception:
            pass

    return list(text)  # character-level fallback


def thai_available() -> bool:
    return _THAI_LIB != "none"