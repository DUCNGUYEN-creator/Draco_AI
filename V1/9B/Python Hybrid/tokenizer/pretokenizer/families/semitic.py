# DracoAI V1 — tokenizer/pretokenizer/families/semitic.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Semitic-script family pre-tokenizer (Arabic, Hebrew).

Arabic/Hebrew are right-to-left but logically stored LTR in Unicode.
Words are whitespace-delimited.  Presentation forms are normalised
upstream (shaping.py).  Harakat (Arabic diacritics) are retained since
they carry phonological information the model may learn from.
"""

import re
from typing import List

try:
    import regex as _re
    _SEMITIC_PATTERN = _re.compile(
        r"(?:\p{L}\p{M}*)+"
        r"|\p{N}{1,3}"
        r"|[^\s\p{L}\p{N}]++"
        r"|\s+",
    )
except ImportError:
    _SEMITIC_PATTERN = re.compile(
        r"(?:[^\W\d_][\u0590-\u06FF\u0750-\u077F\u0300-\u036F]*)+"
        r"|\d{1,3}"
        r"|\s+"
        r"|[^\w\s]",
        re.UNICODE,
    )


def split_semitic(text: str) -> List[str]:
    return [m.group() for m in _SEMITIC_PATTERN.finditer(text) if m.group()]
