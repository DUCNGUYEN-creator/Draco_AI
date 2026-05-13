# DracoAI V1 — modeling/runtime/scheduler.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Continuous Batching Scheduler.

FIX-SLOT-RESET  : cache.reset(batch_idx) is called when a request finishes,
                  before the slot is reused for the next request.
FIX-IMPORT-PATH : uses correct kv_cache.kv_cache path (not cache.kv_cache).
"""
from __future__ import annotations
import threading
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..transformer       import DracoTransformerV1
    from ..kv_cache.kv_cache import KVCache

__all__ = ["RequestHandle", "ContinuousBatchingScheduler"]


class RequestHandle:
    def __init__(self, request_id: int, prompt_ids: List[int],
                 max_new_tokens: int, eos_ids: Optional[set] = None):
        self.request_id     = request_id
        self.prompt_ids     = list(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self.eos_ids        = eos_ids or {151645}
        self.generated:     List[int] = []
        self.done:          bool      = False
        self.slot:          int       = -1
        self._pending_cur:  Optional[List[int]] = list(prompt_ids)

    def __repr__(self) -> str:
        return (f"RequestHandle(id={self.request_id}, slot={self.slot}, "
                f"gen={len(self.generated)}/{self.max_new_tokens}, done={self.done})")


class ContinuousBatchingScheduler:
    """
    Simple continuous batching scheduler.

    Each active request occupies one cache slot (batch_idx).
    The scheduler cycles through slots, running one forward per slot per step.
    When a request finishes, the slot is reset and made available for a new one.
    """

    def __init__(self, model: "DracoTransformerV1", cache: "KVCache",
                 max_slots: int, eos_id: int = 151645):
        self._model      = model
        self._cache      = cache
        self._max_slots  = max_slots
        self._eos_id     = eos_id
        self._slots:     List[Optional[RequestHandle]] = [None] * max_slots
        self._queue:     List[RequestHandle]           = []
        self._next_id    = 0
        self._lock       = threading.Lock()
        self._step_count = 0

    def enqueue(self, prompt_ids: List[int], max_new_tokens: int = 128,
                eos_ids: Optional[set] = None) -> RequestHandle:
        with self._lock:
            h = RequestHandle(
                self._next_id, prompt_ids, max_new_tokens,
                eos_ids or {self._eos_id})
            self._next_id += 1
            self._queue.append(h)
            self._try_assign_nolock()
        return h

    def step(self) -> int:
        """
        Run one decode step for every active (non-done) slot.
        Returns the number of active slots processed.
        """
        with self._lock:
            self._try_assign_nolock()
            active = [h for h in self._slots if h is not None]

        n_active = 0
        for h in active:
            if h.done:
                continue
            n_active += 1
            cur = (h._pending_cur
                   or ([h.generated[-1]] if h.generated else [h.prompt_ids[-1]]))
            h._pending_cur = None

            try:
                l1, _l2, _ = self._model.forward(
                    cur, self._cache, batch_idx=h.slot)
            except Exception:
                h.done = True
                continue

            last_logits = np.clip(l1[0, -1].astype(np.float64), -50.0, 50.0)
            probs = np.exp(last_logits - last_logits.max())
            probs /= probs.sum() + 1e-9
            token_id = int(np.random.choice(len(probs), p=probs))
            h.generated.append(token_id)

            if token_id in h.eos_ids or len(h.generated) >= h.max_new_tokens:
                h.done = True
                with self._lock:
                    if h.slot >= 0:
                        # ✅ FIX-SLOT-RESET: zero the cache for this slot
                        self._cache.reset(batch_idx=h.slot)
                        self._slots[h.slot] = None
                        h.slot = -1
                    self._try_assign_nolock()

        self._step_count += 1
        return n_active

    def all_done(self) -> bool:
        with self._lock:
            return (not self._queue
                    and all(s is None or s.done for s in self._slots))

    def status(self) -> dict:
        with self._lock:
            active = sum(1 for s in self._slots if s is not None and not s.done)
            queued = len(self._queue)
            done   = sum(1 for s in self._slots if s is not None and s.done)
        return dict(active_slots=active, queued=queued,
                    done_slots=done, step_count=self._step_count)

    def _try_assign_nolock(self):
        for i, slot in enumerate(self._slots):
            if slot is None and self._queue:
                h = self._queue.pop(0)
                h.slot = i
                self._slots[i] = h

    def __repr__(self) -> str:
        s = self.status()
        return (f"ContinuousBatchingScheduler(slots={self._max_slots}, "
                f"active={s['active_slots']}, queued={s['queued']}, "
                f"steps={s['step_count']})")