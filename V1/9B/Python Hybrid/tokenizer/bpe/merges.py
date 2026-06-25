# DracoAI V1 — tokenizer/bpe/merges.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — BPE Merge Engine
======================================
High-level interface combining MergeRanks with the heap-based BPE loop.
"""

from typing import Dict, List, Optional, Tuple

from .ranks      import MergeRanks
from .heap_merge import bpe_merge


class MergeEngine:
    """Thin coordinator: holds merge rules, exposes encode()."""

    def __init__(self) -> None:
        self._ranks = MergeRanks()

    def add_merge(self, pair: Tuple[int, int], merged_id: int) -> int:
        return self._ranks.add(pair, merged_id)

    def load(self, rules: List[Tuple[Tuple[int, int], int]]) -> None:
        self._ranks.clear()
        self._ranks.bulk_load(rules)

    def clear(self) -> None:
        self._ranks.clear()

    def encode(self, byte_ids: List[int]) -> List[int]:
        """Apply BPE merges to *byte_ids* and return merged token IDs."""
        return bpe_merge(byte_ids, self._ranks.rank, self._ranks.merges)

    # ── Inspection ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._ranks)

    def __contains__(self, pair: object) -> bool:
        return pair in self._ranks

    def get_rank(self, pair: Tuple[int, int]) -> Optional[int]:
        return self._ranks.get_rank(pair)

    def get_merged_id(self, pair: Tuple[int, int]) -> Optional[int]:
        return self._ranks.get_merged_id(pair)

    @property
    def merges(self) -> Dict[Tuple[int, int], int]:
        return dict(self._ranks.merges)

    @property
    def merge_rank(self) -> Dict[Tuple[int, int], int]:
        return dict(self._ranks.rank)
