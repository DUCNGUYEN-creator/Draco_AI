# DracoAI V1 — modeling/kv_cache/snapshot.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .kv_cache import KVCache

__all__ = ["SnapshotStack"]


class SnapshotStack:
    """
    Nestable snapshot manager for speculative / beam rollback.

    push() → saves current cache state.
    pop()  → restores and returns the most recent snapshot.
    commit() → discards the most recent snapshot without restoring.
    rollback_to(level) → restores all snapshots above `level`.
    """

    def __init__(self, cache: "KVCache"):
        self._cache  = cache
        self._snaps: List[dict] = []

    def push(self, batch_idx: int = 0, delta_threshold: int = 64) -> dict:
        snap = self._cache.snapshot(delta_threshold=delta_threshold, batch_idx=batch_idx)
        snap["_stack_level"]    = len(self._snaps)
        snap["_parent_snap_id"] = id(self._snaps[-1]) if self._snaps else None
        self._snaps.append(snap)
        return snap

    def pop(self) -> Optional[dict]:
        if not self._snaps:
            return None
        snap = self._snaps.pop()
        self._cache.restore(snap)
        return snap

    def commit(self) -> Optional[dict]:
        """Accept the top snapshot (no restore) — discard the saved state."""
        return self._snaps.pop() if self._snaps else None

    def rollback_to(self, level: int):
        """Restore every snapshot above `level`."""
        while len(self._snaps) > level:
            self._cache.restore(self._snaps.pop())

    def clear(self):
        self._snaps.clear()

    @property
    def depth(self) -> int:
        return len(self._snaps)

    def __repr__(self) -> str:
        return f"SnapshotStack(depth={self.depth})"