# DracoAI V1 — modeling/kv_cache/prefix_cache.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""LRU prompt prefix cache.

FIXES (this revision):
  ✅ FIX-PREFIX-ENGRAM-SNAPSHOT  : put() stores an optional engram_snap.
     get() returns it as the 4th element.
  ✅ FIX-PREFIX-TUPLE-UNPACK     : get() always returns a 4-tuple
     (snap, plen, last_logits, engram_snap).  Legacy 3/4-element entries are
     padded with None so callers never see a variable-length return.
  ✅ FIX-LRU-EVICTION-EFFICIENCY : OrderedDict + move_to_end / popitem.
  ✅ FIX-STORE-TUPLE-CONSISTENT  : The internal store always uses 5-tuples
     (snap, plen, ts, last_logits, engram_snap) so len(entry) checks are
     deterministic and put() overwrites are safe.
"""
from __future__ import annotations

import collections
import hashlib
import struct
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from ..constants import ROPE_THETA

__all__ = ["PrefixCache"]


class PrefixCache:
    """
    Thread-safe LRU cache keyed on (token_id_sequence, rope_theta).

    get() always returns a 4-tuple:
        (snap, prefix_len, last_logits, engram_snap)
    or None on miss.

    put() stores a full KV snapshot plus optional last logit vector and
    optional Engram snapshot.
    """

    def __init__(self, max_entries: int = 32):
        self._max   = max_entries
        self._store: collections.OrderedDict = collections.OrderedDict()
        self._lock  = threading.Lock()

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _hash(token_ids: List[int], rope_theta: float = ROPE_THETA) -> str:
        buf  = np.array(token_ids, dtype=np.int32).tobytes()
        buf += struct.pack("<d", float(rope_theta))
        return hashlib.sha256(buf).hexdigest()

    # ── Public API ────────────────────────────────────────────────────────────

    def get(
        self,
        prefix_ids: List[int],
        rope_theta: float = ROPE_THETA,
    ) -> Optional[Tuple[dict, int, Optional[np.ndarray], Optional[dict]]]:
        """
        Return (snap, prefix_len, last_logits, engram_snap) on hit, else None.

        engram_snap is None for entries stored without Engram support.
        last_logits is None for entries stored without a logit vector.

        ✅ FIX-PREFIX-TUPLE-UNPACK: always returns a 4-tuple by normalising
        legacy 3-tuple and 4-tuple entries stored before the current schema.
        """
        h = self._hash(prefix_ids, rope_theta)
        with self._lock:
            if h not in self._store:
                return None
            entry = self._store[h]

            # Normalise to 5-tuple regardless of legacy storage format
            if len(entry) == 5:
                snap, plen, _, last_logits, engram_snap = entry
            elif len(entry) == 4:
                # Legacy: (snap, plen, ts, last_logits)
                snap, plen, _, last_logits = entry
                engram_snap = None
            else:
                # Very old 3-tuple: (snap, plen, ts)
                snap, plen, _ = entry
                last_logits  = None
                engram_snap  = None

            # O(1) LRU refresh
            self._store.move_to_end(h)
            # Rewrite as canonical 5-tuple with refreshed timestamp
            self._store[h] = (snap, plen, time.perf_counter(),
                              last_logits, engram_snap)
            return snap, plen, last_logits, engram_snap

    def put(
        self,
        prefix_ids:  List[int],
        snap:        dict,
        last_logits: Optional[np.ndarray] = None,
        rope_theta:  float = ROPE_THETA,
        engram_snap: Optional[dict] = None,
    ):
        """
        Store a prefix cache entry.

        Parameters
        ──────────
        snap        : full KV cache snapshot (from cache.snapshot()).
        last_logits : logit vector for the last prompt token.
        engram_snap : lightweight Engram snapshot (from engram.snapshot()).
        """
        h = self._hash(prefix_ids, rope_theta)
        with self._lock:
            if h in self._store:
                self._store[h] = (snap, len(prefix_ids), time.perf_counter(),
                                  last_logits, engram_snap)
                self._store.move_to_end(h)
                return
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[h] = (snap, len(prefix_ids), time.perf_counter(),
                              last_logits, engram_snap)

    def invalidate(self, prefix_ids: List[int], rope_theta: float = ROPE_THETA):
        with self._lock:
            self._store.pop(self._hash(prefix_ids, rope_theta), None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"PrefixCache(entries={len(self)}/{self._max})"