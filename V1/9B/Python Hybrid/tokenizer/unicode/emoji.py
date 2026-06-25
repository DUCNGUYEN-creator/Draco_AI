# DracoAI V1 — tokenizer/unicode/emoji.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Emoji & ZWJ Sequence Handling
==================================================
Detects and anchors emoji sequences as indivisible atoms before BPE.

Handles:
  - Skin-tone modifiers (👍🏽 = base + Fitzpatrick U+1F3FB–1F3FF)
  - ZWJ families (👨‍👩‍👧‍👦 = multiple bases joined by U+200D)
  - Regional flag pairs (🇻🇳 = two regional indicator symbols)
  - Keycap sequences (#️⃣ = digit + VS16 + U+20E3)
  - Tag flags (🏴󠁧󠁢󠁥󠁮󠁧󠁿 = base + tag chars + cancel tag U+E007F)
  - Variation selectors (☎️ = base + U+FE0F)

Uses the ``regex`` library when available for EMOJI_PATTERN (which
requires Unicode property support).  Falls back to a hand-rolled
implementation using unicodedata when regex is not installed.
"""

import re
import unicodedata
from typing import Iterator, List, Tuple

from ..constants import (
    EMOJI_RANGES, ZWJ, FITZPATRICK_START, FITZPATRICK_END,
    VARIATION_SELECTOR_START, VARIATION_SELECTOR_END,
    REGIONAL_INDICATOR_START, REGIONAL_INDICATOR_END,
    COMBINING_ENCLOSING_KEYCAP, TAG_START, TAG_END, CANCEL_TAG,
)

# ── Try regex for production-grade emoji pattern ──────────────────────
try:
    import regex as _re
    _HAS_REGEX = True
    # SOTA ZWJ-aware emoji pattern (Unicode Emoji 15.0+)
    _EMOJI_PATTERN = _re.compile(
        r"""(
            # Type 1: Subdivision tag flags (🏴󠁧󠁢󠁥󠁮󠁧󠁿)
            (?:[\U0001F3F4]|🏴)[\u200D]?[\U000E0020-\U000E007F]{2,6}[\U000E007F]
            |
            # Type 2: ZWJ sequences with optional Fitzpatrick modifiers
            (?:
                (?:[\u2600-\u27BF]|[\U0001F000-\U0001FFFF])
                (?:[\U0001F3FB-\U0001F3FF])?
                (?:\uFE0F)?
            )
            (?:
                \u200D
                (?:[\u2600-\u27BF]|[\U0001F000-\U0001FFFF])
                (?:[\U0001F3FB-\U0001F3FF])?
                (?:\uFE0F)?
            )*
            |
            # Type 3: Regional indicator pairs (national flags 🇻🇳)
            [\U0001F1E6-\U0001F1FF]{2}
            |
            # Type 4: Keycap sequences (#️⃣)
            [\u0023-\u0039]\uFE0F?\u20E3
        )""",
        _re.VERBOSE,
    )
except ImportError:
    _HAS_REGEX = False
    _EMOJI_PATTERN = None  # type: ignore[assignment]


# ── Predicates ───────────────────────────────────────────────────────

def is_emoji_base(c: str) -> bool:
    cp = ord(c)
    return any(lo <= cp <= hi for lo, hi in EMOJI_RANGES)


def is_fitzpatrick_modifier(c: str) -> bool:
    return FITZPATRICK_START <= ord(c) <= FITZPATRICK_END


def is_regional_indicator(c: str) -> bool:
    return REGIONAL_INDICATOR_START <= ord(c) <= REGIONAL_INDICATOR_END


def is_variation_selector(c: str) -> bool:
    cp = ord(c)
    return ((VARIATION_SELECTOR_START <= cp <= VARIATION_SELECTOR_END) or
            (0xE0100 <= cp <= 0xE01EF))


def is_tag_char(c: str) -> bool:
    return TAG_START <= ord(c) <= TAG_END


def is_zwj(c: str) -> bool:
    return c == ZWJ


def is_keycap(c: str) -> bool:
    return c == COMBINING_ENCLOSING_KEYCAP


def contains_emoji(text: str) -> bool:
    return any(is_emoji_base(c) for c in text)


def emoji_count(text: str) -> int:
    return sum(1 for _ in iter_emoji_sequences(text))


# ── Fallback hand-rolled emoji sequence iterator ──────────────────────

def _fallback_iter_sequences(text: str) -> Iterator[Tuple[int, int, str]]:
    """Hand-rolled emoji sequence iterator (no regex dep)."""
    chars = list(text)
    n     = len(chars)
    i     = 0

    while i < n:
        c = chars[i]

        # Regional pair → flag
        if is_regional_indicator(c):
            if i + 1 < n and is_regional_indicator(chars[i + 1]):
                start = i
                seq   = c + chars[i + 1]
                i    += 2
                yield (start, i, seq)
            else:
                i += 1
            continue

        if not is_emoji_base(c):
            i += 1
            continue

        start = i
        seq   = c
        i    += 1

        # Variation selector / VS16
        while i < n and is_variation_selector(chars[i]):
            seq += chars[i]
            i   += 1

        # Keycap
        if i < n and is_keycap(chars[i]):
            seq += chars[i]
            i   += 1

        # Fitzpatrick modifier
        if i < n and is_fitzpatrick_modifier(chars[i]):
            seq += chars[i]
            i   += 1

        # ZWJ chain
        while i < n and is_zwj(chars[i]) and i + 1 < n:
            seq += chars[i]
            i   += 1
            seq += chars[i]
            i   += 1
            while i < n and (is_fitzpatrick_modifier(chars[i]) or
                              is_variation_selector(chars[i])):
                seq += chars[i]
                i   += 1

        # Tag sequence
        while i < n and is_tag_char(chars[i]):
            seq += chars[i]
            i   += 1
        if i < n and chars[i] == CANCEL_TAG:
            seq += chars[i]
            i   += 1

        yield (start, i, seq)


def iter_emoji_sequences(text: str) -> Iterator[Tuple[int, int, str]]:
    """
    Iterate over emoji sequences, yielding (start, end, sequence) tuples.
    Uses regex library when available for correctness.
    """
    if _HAS_REGEX and _EMOJI_PATTERN is not None:
        for m in _EMOJI_PATTERN.finditer(text):
            yield (m.start(), m.end(), m.group())
    else:
        yield from _fallback_iter_sequences(text)


def split_preserving_emoji(text: str) -> List[str]:
    """
    Split *text* into segments, keeping every emoji ZWJ sequence
    as one indivisible string atom.

    Non-emoji text between sequences is included as separate segments.
    """
    if _HAS_REGEX and _EMOJI_PATTERN is not None:
        parts = _EMOJI_PATTERN.split(text)
        return [p for p in parts if p]

    # Fallback: manual reconstruction
    result: List[str] = []
    prev = 0
    for start, end, seq in _fallback_iter_sequences(text):
        if start > prev:
            result.append(text[prev:start])
        result.append(seq)
        prev = end
    if prev < len(text):
        result.append(text[prev:])
    return result
