# DracoAI V1 — tokenizer/bpe/ranks.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""DracoAI Tokenizer — BPE merge rank table."""

from typing import Dict, Iterator, List, Optional, Tuple


class MergeRanks:
    """Bidirectional merge rule table: pair ↔ (rank, merged_id)."""

    __slots__ = ("merges", "rank")

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.rank:   Dict[Tuple[int, int], int] = {}

    def add(self, pair: Tuple[int, int], merged_id: int) -> int:
        """Register a merge rule. Returns the assigned rank.

        Idempotent for rank: if *pair* already has a rank, the existing
        rank is kept (rank order never changes after first registration).
        The merged_id is always updated to the latest value provided.
        """
        if pair not in self.rank:
            self.rank[pair] = len(self.rank)
        self.merges[pair] = merged_id
        return self.rank[pair]

    def bulk_load(self, rules: List[Tuple[Tuple[int, int], int]]) -> None:
        for pair, merged_id in rules:
            self.add(pair, merged_id)

    def get_rank(self, pair: Tuple[int, int]) -> Optional[int]:
        return self.rank.get(pair)

    def get_merged_id(self, pair: Tuple[int, int]) -> Optional[int]:
        return self.merges.get(pair)

    def items(self) -> Iterator[Tuple[Tuple[int, int], int]]:
        for pair in sorted(self.rank, key=lambda p: self.rank[p]):
            yield pair, self.merges[pair]

    def clear(self) -> None:
        self.merges.clear()
        self.rank.clear()

    def __len__(self) -> int:
        return len(self.merges)

    def __contains__(self, pair: object) -> bool:
        return pair in self.merges