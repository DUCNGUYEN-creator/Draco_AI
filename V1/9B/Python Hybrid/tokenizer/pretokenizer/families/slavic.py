# DracoAI V1 — tokenizer/pretokenizer/families/slavic.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Slavic-script family pre-tokenizer (Russian, Ukrainian, Bulgarian,
Serbian, Belarusian, Macedonian — all Cyrillic-based).

Uses the same SOTA Latin pattern since Cyrillic word structure is
similar: whitespace-delimited words, punctuation isolated.
"""

import re
from typing import List

try:
    import regex as _re
    _SLAVIC_PATTERN = _re.compile(
        r"[^\r\n\p{L}\p{N}]?(?:\p{L}\p{M}*)+"
        r"|\p{N}{1,3}"
        r"|[^\s\p{L}\p{N}]++"
        r"|[\r\n]+"
        r"|\s+",
        _re.VERBOSE,
    )
except ImportError:
    _SLAVIC_PATTERN = re.compile(
        r"(?:[^\W\d_][\u0300-\u036F]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def split_slavic(text: str) -> List[str]:
    return [m.group() for m in _SLAVIC_PATTERN.finditer(text) if m.group()]
