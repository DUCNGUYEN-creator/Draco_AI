# DracoAI V1 — tokenizer/unicode/normalize.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Unicode Normalisation
==========================================
NFC / NFKC / NFD / NFKD normalisation helpers.

Vietnamese note: 'ắ' can be stored as one precomposed codepoint (NFC)
or as 'a' + U+0306 (breve) + U+0301 (acute) (NFD).  Forcing NFC before
BPE ensures both representations produce the same token sequence.
"""

import unicodedata
from typing import FrozenSet

_VALID_FORMS: FrozenSet[str] = frozenset({"NFC", "NFKC", "NFD", "NFKD"})


def normalize(text: str, form: str = "NFC") -> str:
    """Normalise *text* to the given Unicode form (NFC / NFKC / NFD / NFKD)."""
    if form not in _VALID_FORMS:
        raise ValueError(f"Invalid normalisation form {form!r}. Choose from {_VALID_FORMS}")
    return unicodedata.normalize(form, text)


def nfc(text: str) -> str:
    """NFC – precomposed (default for DracoAI tokenizer)."""
    return unicodedata.normalize("NFC", text)


def nfkc(text: str) -> str:
    """NFKC – compatibility + composition (folds fullwidth, ligatures, etc.)."""
    return unicodedata.normalize("NFKC", text)


def nfd(text: str) -> str:
    """NFD – canonical decomposed."""
    return unicodedata.normalize("NFD", text)


def nfkd(text: str) -> str:
    """NFKD – compatibility + decomposed."""
    return unicodedata.normalize("NFKD", text)


def is_nfc(text: str) -> bool:
    """Return True if *text* is already in NFC form."""
    return unicodedata.is_normalized("NFC", text)


def strip_combining_marks(text: str) -> str:
    """Remove all Unicode combining marks (category Mn) – for analysis only."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def fold_fullwidth(text: str) -> str:
    """Fold fullwidth ASCII / halfwidth Katakana to standard forms via NFKC."""
    return unicodedata.normalize("NFKC", text)