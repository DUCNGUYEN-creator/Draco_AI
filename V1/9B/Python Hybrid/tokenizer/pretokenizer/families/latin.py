# DracoAI V1 — tokenizer/pretokenizer/families/latin.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Latin-script family pre-tokenizer (English, French, Spanish, Portuguese,
German, Italian, Dutch, Romanian, Vietnamese, ...).

SOTA Master Regex Pattern (GPT-4 / Qwen-compatible):
  - English contractions: 's, 't, 're, 've, 'm, 'll, 'd
  - \p{L}\p{M}*  protects Vietnamese diacritics (letter + combining marks)
  - \p{N}{1,3}   Digit tri-partitioning (prevents number-token bloat)
  - Punctuation isolation
  - Whitespace handling with leading-space prefix (GPT-style)
"""

import re
from typing import List

# Try ``regex`` for Unicode property support (\p{L}, \p{M}, \p{N})
try:
    import regex as _re
    _HAS_REGEX = True
    # SOTA pattern — mirrors GPT-4 / Qwen pre-tokenizer, adds diacritic guard
    _LATIN_PATTERN = _re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""    # English contractions
        r"""|[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+"""  # words with combining marks
        r"""|\p{N}{1,3}"""                       # numbers <=3 digits (tri-partition)
        r"""|[^\s\p{L}\p{N}]++"""                # punctuation / symbols run
        r"""|[\r\n]+"""                           # line breaks
        r"""|\s+(?!\S)"""                         # trailing whitespace (code indent)
        r"""|\s+""",                              # other whitespace
        _re.VERBOSE,
    )
except ImportError:
    _HAS_REGEX = False
    # Fallback regex (re module) — no \p{} properties, adequate for most Latin text
    _LATIN_PATTERN = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|(?:[^\W\d_][\u0300-\u036F]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def split_latin(text: str) -> List[str]:
    """
    Segment a Latin-script text segment into pre-tokenizer units.

    Parameters
    ----------
    text : str
        NFC-normalised text, special tokens already removed.

    Returns
    -------
    List[str]
        Non-empty segments ready for UTF-8 byte encoding and BPE.
    """
    return [m.group() for m in _LATIN_PATTERN.finditer(text) if m.group()]
