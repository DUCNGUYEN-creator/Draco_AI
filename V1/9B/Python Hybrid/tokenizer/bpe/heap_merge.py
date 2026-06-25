# DracoAI V1 — tokenizer/bpe/heap_merge.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — O(N log N) BPE Heap Merge Engine
=====================================================
Applies BPE merge rules to a byte-ID sequence using a min-heap.

Algorithm
---------
1. Build a doubly-linked list over the input sequence.
2. Push all adjacent pairs onto a min-heap keyed by (rank, counter).
   Monotonic counter breaks ties deterministically.
3. Pop the lowest-rank pair; validate liveness and adjacency.
4. Merge: replace left with merged_id, unlink right.
5. Push new adjacent pairs; repeat until heap empty.

Complexity: O(N log N) time, O(N) space.
"""

import heapq
from typing import Dict, List, Tuple


def bpe_merge(
    byte_ids:    List[int],
    merge_rank:  Dict[Tuple[int, int], int],
    merges:      Dict[Tuple[int, int], int],
) -> List[int]:
    """
    Apply BPE merges to *byte_ids*.

    Parameters
    ----------
    byte_ids : List[int]
        Initial token IDs (byte-level, 0–255).
    merge_rank : Dict[Tuple[int,int], int]
        (left, right) → priority rank (lower = earlier merge).
    merges : Dict[Tuple[int,int], int]
        (left, right) → merged token ID.

    Returns
    -------
    List[int]
        Token IDs after all applicable BPE merges.
    """
    n = len(byte_ids)
    if n == 0:
        return []
    if n == 1:
        return byte_ids[:]

    ids    = list(byte_ids)
    active = [True] * n
    prev   = list(range(-1, n - 1))   # doubly-linked list prev pointers
    nxt    = list(range(1, n + 1))    # next pointers
    nxt[n - 1] = n                    # sentinel

    heap: list = []
    counter    = 0   # monotonic tie-breaker

    def push(li: int, ri: int) -> None:
        nonlocal counter
        pair = (ids[li], ids[ri])
        rank = merge_rank.get(pair)
        if rank is not None:
            heapq.heappush(heap, (rank, counter, li, ri))
            counter += 1

    for i in range(n - 1):
        push(i, i + 1)

    while heap:
        rank, _, li, ri = heapq.heappop(heap)

        # Stale-entry validation
        if not active[li] or not active[ri]:
            continue
        if nxt[li] != ri:
            continue
        pair = (ids[li], ids[ri])
        if merge_rank.get(pair) != rank:
            continue

        # Apply merge
        ids[li] = merges[pair]
        active[ri] = False
        rri        = nxt[ri]
        nxt[li]    = rri
        if rri < n:
            prev[rri] = li

        # Push new neighbouring pairs
        lli = prev[li]
        if lli >= 0 and active[lli]:
            push(lli, li)
        if rri < n and active[rri]:
            push(li, rri)

    return [ids[i] for i in range(n) if active[i]]