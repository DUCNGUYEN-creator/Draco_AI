# DracoAI V1 — tokenizer/constants.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
DracoAI Tokenizer — Constants
==============================
Single source of truth for all special tokens, Unicode constants,
script ranges, and global config values used across the tokenizer package.
"""

from typing import Dict, FrozenSet, List, Tuple

# ── Special tokens (Qwen 3.5 9B ChatML style) ────────────────────────
SPECIAL_TOKENS: Dict[str, int] = {
    "<|endoftext|>": 151643,
    "<|im_start|>":  151644,
    "<|im_end|>":    151645,
    "<|pad|>":       151646,
    "<|unk|>":       151647,
    "<|sep|>":       151648,
    "<think>":       151649,
    "</think>":      151650,
    "<tool_call>":   151651,
    "</tool_call>":  151652,
}

# Reverse lookup: id → name (built once at import time)
SPECIAL_ID_TO_NAME: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

N_SPECIAL     = len(SPECIAL_TOKENS)
QWEN_BASE_END = 151936  # Qwen 3.5 9B vocab boundary — extension IDs start here

BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"

# ── Stream-flush trigger characters ──────────────────────────────────
FLUSH_CHARS: FrozenSet[str] = frozenset(
    " \t\n.,!?;:—–()[]{}'\"，。！？；："
)

# ── Vietnamese diacritic character set (precomposed NFC) ─────────────
VIET_CHARS: FrozenSet[str] = frozenset(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ"
)

# ── Unicode script ranges ─────────────────────────────────────────────
CJK_RANGES: List[Tuple[int, int]] = [
    (0x4E00,  0x9FFF),    # CJK Unified Ideographs
    (0x3400,  0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),   # CJK Extension B
    (0xF900,  0xFAFF),    # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),   # CJK Compatibility Supplement
]

HIRAGANA_RANGE   = (0x3040, 0x309F)
KATAKANA_RANGE   = (0x30A0, 0x30FF)
HANGUL_RANGE     = (0xAC00, 0xD7AF)
HANGUL_JAMO      = (0x1100, 0x11FF)
ARABIC_RANGE     = (0x0600, 0x06FF)
CYRILLIC_RANGE   = (0x0400, 0x04FF)
THAI_RANGE       = (0x0E00, 0x0E7F)
LAO_RANGE        = (0x0E80, 0x0EFF)
KHMER_RANGE      = (0x1780, 0x17FF)
MYANMAR_RANGE    = (0x1000, 0x109F)
TIBETAN_RANGE    = (0x0F00, 0x0FFF)
DEVANAGARI_RANGE = (0x0900, 0x097F)
HEBREW_RANGE     = (0x0590, 0x05FF)
GEORGIAN_RANGE   = (0x10A0, 0x10FF)
ARMENIAN_RANGE   = (0x0530, 0x058F)
ETHIOPIC_RANGE   = (0x1200, 0x137F)

# ── Emoji Unicode ranges ──────────────────────────────────────────────
EMOJI_RANGES: List[Tuple[int, int]] = [
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F300, 0x1F5FF),  # Misc symbols and pictographs
    (0x1F680, 0x1F6FF),  # Transport and map symbols
    (0x1F700, 0x1F77F),  # Alchemical symbols
    (0x1F780, 0x1F7FF),  # Geometric shapes extended
    (0x1F800, 0x1F8FF),  # Supplemental arrows C
    (0x1F900, 0x1F9FF),  # Supplemental symbols and pictographs
    (0x1FA00, 0x1FA6F),  # Chess symbols
    (0x1FA70, 0x1FAFF),  # Symbols and pictographs extended A
    (0x2600,  0x26FF),   # Misc symbols
    (0x2700,  0x27BF),   # Dingbats
    (0xFE00,  0xFE0F),   # Variation selectors
    (0x1F1E0, 0x1F1FF),  # Regional indicator symbols (flags)
]

# ── Zero-Width Joiner and special combining codepoints ───────────────
ZWJ                       = "\u200D"  # Zero-Width Joiner
VARIATION_SELECTOR_16     = "\uFE0F"  # Emoji presentation selector
COMBINING_ENCLOSING_KEYCAP = "\u20E3"
CANCEL_TAG                = "\U000E007F"

VARIATION_SELECTOR_START  = 0xFE00
VARIATION_SELECTOR_END    = 0xFE0F
FITZPATRICK_START         = 0x1F3FB
FITZPATRICK_END           = 0x1F3FF
REGIONAL_INDICATOR_START  = 0x1F1E0
REGIONAL_INDICATOR_END    = 0x1F1FF
TAG_START                 = 0xE0000
TAG_END                   = 0xE007F

# ── Scriptio Continua scripts (no whitespace word boundaries) ────────
# These require character-level splitting in the pre-tokenizer
SCRIPTIO_CONTINUA_RANGES: List[Tuple[int, int]] = [
    THAI_RANGE,
    LAO_RANGE,
    KHMER_RANGE,
    MYANMAR_RANGE,
    TIBETAN_RANGE,
]

# ── Security limits ───────────────────────────────────────────────────
MAX_INPUT_LENGTH       = 200_000   # chars before truncation
MAX_CONSECUTIVE_SPACES = 32        # spam-space DoS protection
MAX_COMBINING_MARKS    = 16        # Zalgo text protection
MAX_BUFFER_BYTES       = 32        # streaming byte-buffer overflow guard