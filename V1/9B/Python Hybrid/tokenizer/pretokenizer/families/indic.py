# DracoAI V1 — tokenizer/pretokenizer/families/indic.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Indic-script family pre-tokenizer (Devanagari: Hindi, Sanskrit, Marathi,
Nepali; also Bengali, Gujarati, Punjabi, Tamil, Telugu, Kannada, etc.).

Indic scripts use space-delimited words (unlike Thai/Khmer).  Each
word is kept intact; splitting is whitespace+punctuation based.
Combining matras (U+0900-U+097F) are treated as part of the preceding
consonant/vowel via \p{L}\p{M}* in the regex.
"""

import re
from typing import List

try:
    import regex as _re
    _INDIC_PATTERN = _re.compile(
        r"(?:\p{L}\p{M}*)+"       # letters + combining marks (matras)
        r"|\p{N}{1,3}"
        r"|[^\s\p{L}\p{N}]++"
        r"|\s+",
    )
except ImportError:
    _INDIC_PATTERN = re.compile(
        r"(?:[^\W\d_][\u0300-\u036F\u0900-\u0D7F\u1C00-\u1CFF]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def split_indic(text: str) -> List[str]:
    return [m.group() for m in _INDIC_PATTERN.finditer(text) if m.group()]
