# DracoAI V1 — tokenizer/pretokenizer/languages/vietnamese.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
r"""
Vietnamese-specific pre-tokenizer.

Vietnamese is Latin-script but with heavy use of precomposed diacritics.
After NFC normalisation (done in encode()) each syllable is a single
codepoint sequence — the SOTA regex pattern \p{L}\p{M}* already handles
this correctly.

This module adds:
  1. Syllable-awareness: Vietnamese words are monosyllabic or
     disyllabic.  We keep the standard Latin splitter (one word per
     segment) since BPE learns syllable merges naturally from data.
  2. Nói lái / đảo ngữ safety: no special transform needed — as long
     as NFC normalisation is applied before splitting, each Vietnamese
     syllable (e.g. "chào", "tiền") remains an indivisible segment
     entering BPE, so the model sees intact syllable tokens and can
     learn transposition patterns.
"""

from typing import List
from ..families.latin import split_latin


def split_vietnamese(text: str) -> List[str]:
    """
    Segment Vietnamese text.

    Delegates to the Latin-family splitter (which uses the SOTA regex
    with \\p{L}\\p{M}* diacritic-binding guard).

    Parameters
    ----------
    text : str
        NFC-normalised Vietnamese text.

    Returns
    -------
    List[str]
        Pre-tokenized segments.
    """
    return split_latin(text)
