# DracoAI V1 — tokenizer/unicode/grapheme.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — Grapheme Cluster Segmentation
==================================================
Splits text into user-perceived characters (grapheme clusters) so
streaming decode always yields visually complete units.

Priority:
  1. ``regex`` library (pip install regex) — full Unicode 15 grapheme
     cluster support via \\X.  Best for production.
  2. Fallback: hand-rolled iterator using unicodedata.

is_safe_to_yield() checks whether the *last character* of the
accumulated text is in a "pending" combining state — i.e. one that
may be extended by the next arriving token.
"""

import unicodedata
from typing import Iterator, List

from ..constants import (
    ZWJ, FITZPATRICK_START, FITZPATRICK_END,
    VARIATION_SELECTOR_START, VARIATION_SELECTOR_END,
    REGIONAL_INDICATOR_START, REGIONAL_INDICATOR_END,
    COMBINING_ENCLOSING_KEYCAP, TAG_START, TAG_END, CANCEL_TAG,
)

# Try regex library for full Unicode grapheme cluster support
try:
    import regex as _regex
    _HAS_REGEX = True
except ImportError:
    _HAS_REGEX = False


# ── Predicates ───────────────────────────────────────────────────────

def _is_combining(c: str) -> bool:
    return unicodedata.category(c) in ("Mn", "Mc", "Me")


def _is_regional_indicator(c: str) -> bool:
    return REGIONAL_INDICATOR_START <= ord(c) <= REGIONAL_INDICATOR_END


def _is_fitzpatrick(c: str) -> bool:
    return FITZPATRICK_START <= ord(c) <= FITZPATRICK_END


def _is_variation_selector(c: str) -> bool:
    cp = ord(c)
    return ((VARIATION_SELECTOR_START <= cp <= VARIATION_SELECTOR_END) or
            (0xE0100 <= cp <= 0xE01EF))


def _is_tag_char(c: str) -> bool:
    return TAG_START <= ord(c) <= TAG_END


def _is_zwj(c: str) -> bool:
    return c == ZWJ


def _is_keycap(c: str) -> bool:
    return c == COMBINING_ENCLOSING_KEYCAP


# ── Fallback grapheme iterator ────────────────────────────────────────

def _fallback_iter_graphemes(text: str) -> Iterator[str]:
    """
    Hand-rolled grapheme cluster iterator.

    Handles: combining marks, ZWJ emoji sequences, regional indicator
    flag pairs, Fitzpatrick modifiers, variation selectors, keycap
    sequences, and Unicode tag sequences (subdivision flags).
    """
    if not text:
        return

    chars = list(text)
    i = 0
    n = len(chars)

    while i < n:
        cluster = chars[i]
        i += 1

        # Regional indicator pair -> flag emoji (e.g. 🇻🇳)
        if _is_regional_indicator(cluster):
            if i < n and _is_regional_indicator(chars[i]):
                cluster += chars[i]
                i += 1
            yield cluster
            continue

        # Variation selector on simple base char
        while i < n and _is_variation_selector(chars[i]):
            cluster += chars[i]
            i += 1

        # Keycap sequence (#️⃣)
        if i < n and _is_keycap(chars[i]):
            cluster += chars[i]
            i += 1

        # Fitzpatrick skin-tone modifier
        if i < n and _is_fitzpatrick(chars[i]):
            cluster += chars[i]
            i += 1

        # ZWJ chain — absorb as long as ZWJ+next_base pattern continues
        while i < n and _is_zwj(chars[i]) and i + 1 < n:
            cluster += chars[i]       # ZWJ
            i += 1
            cluster += chars[i]       # next base character
            i += 1
            # modifiers after the ZWJ-joined base
            while i < n and (_is_fitzpatrick(chars[i]) or
                              _is_variation_selector(chars[i])):
                cluster += chars[i]
                i += 1

        # Tag sequence (subdivision flags like 🏴󠁧󠁢󠁥󠁮󠁧󠁿)
        while i < n and _is_tag_char(chars[i]):
            cluster += chars[i]
            i += 1
        if i < n and chars[i] == CANCEL_TAG:
            cluster += chars[i]
            i += 1

        # Remaining combining marks / diacritics
        while i < n and _is_combining(chars[i]):
            cluster += chars[i]
            i += 1

        yield cluster


def _regex_iter_graphemes(text: str) -> Iterator[str]:
    """Use regex library \\X for full Unicode 15 grapheme cluster support."""
    for m in _regex.finditer(r"\X", text):
        yield m.group()


def iter_graphemes(text: str) -> Iterator[str]:
    """
    Iterate over Unicode grapheme clusters in *text*.

    Uses ``regex`` when available; falls back to the hand-rolled impl.
    """
    if _HAS_REGEX:
        yield from _regex_iter_graphemes(text)
    else:
        yield from _fallback_iter_graphemes(text)


def split_graphemes(text: str) -> List[str]:
    """Return a list of grapheme clusters for *text*."""
    return list(iter_graphemes(text))


def grapheme_len(text: str) -> int:
    """Return the number of grapheme clusters in *text*."""
    return sum(1 for _ in iter_graphemes(text))


# ── Safe-yield helpers ────────────────────────────────────────────────

def is_safe_to_yield(text: str) -> bool:
    """
    Return True if *text* ends on a complete grapheme cluster boundary.

    Checks whether the last codepoint of *text* is a "pending" character
    — i.e., one that may be extended by the next arriving token:
    combining diacritic, ZWJ, Fitzpatrick modifier, variation selector,
    unpaired regional indicator, or Unicode tag character.

    Parameters
    ----------
    text : str
        Accumulated output text, potentially mid-grapheme.

    Returns
    -------
    bool
        False if the last codepoint is pending (must wait for more tokens).
        True if it is safe to flush *text* to the UI.
    """
    if not text:
        return True

    last = text[-1]

    # ZWJ at the end -> next token will likely be another emoji base
    if _is_zwj(last):
        return False

    # Fitzpatrick modifier: may be followed by more ZWJ
    if _is_fitzpatrick(last):
        return False

    # Variation selector: the character gained a presentation form,
    # but may still be followed by ZWJ or keycap
    if _is_variation_selector(last):
        return False

    # Combining diacritic — definitely pending
    if _is_combining(last):
        return False

    # Regional indicator: safe only when paired (second of the pair)
    if _is_regional_indicator(last):
        # If the previous character is also a regional indicator, this is
        # the second of a complete flag pair -> safe.
        if len(text) < 2 or not _is_regional_indicator(text[-2]):
            return False

    # Unicode tag character — subdivision flag in progress
    if _is_tag_char(last):
        return False

    return True


def safe_flush_point(text: str) -> int:
    """
    Return the character-index up to which *text* can be safely emitted.

    Walks grapheme clusters from the start; returns the cumulative char
    length of all complete clusters, leaving any trailing partial cluster
    buffered.

    Returns len(text) if the entire string is safe to emit, 0 if nothing
    can be safely emitted yet.
    """
    if not text:
        return 0

    # Fast path: the whole string ends on a safe boundary.
    if is_safe_to_yield(text):
        return len(text)

    # Build grapheme list to find the safe prefix.
    graphemes = list(iter_graphemes(text))

    if len(graphemes) <= 1:
        # The entire text is one incomplete cluster — cannot emit anything.
        return 0

    # The last grapheme is pending; sum all but the last.
    return sum(len(g) for g in graphemes[:-1])
