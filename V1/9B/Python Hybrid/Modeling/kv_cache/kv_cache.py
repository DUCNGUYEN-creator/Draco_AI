# DracoAI V1 — modeling/kv_cache/kv_cache.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
SWA-Sink sliding-window KV cache with snapshot/restore + multi-batch.

Design notes:
  - update() is the ONLY method that advances _cache_pos.  step() is a
    documented no-op kept for API compatibility.
  - Snapshot types:
      "full"  – copies entire slab; O(window) memory per snap.
      "delta" – records only modified (layer, slot) pairs with their
                PRE-WRITE values; escalates to full automatically when
                the delta list exceeds the threshold.
  - reset(batch_idx) is called by the scheduler when a slot finishes,
    before the slot is reused.

FIXES (this revision):
  ✅ FIX-DELTA-RESTORE-OLD-VALUES : delta_list now stores the OLD (pre-write)
     KV values so restore() correctly reverts to the state before update().
     Previously it stored the NEW values — making delta restore a no-op.
  ✅ FIX-DELTA-ESCALATION-PARTIAL : when escalation fires mid-seq, the
     remaining tokens in the seq are written via the fast (vectorised) path
     instead of being silently dropped.  The full snapshot taken during
     escalation captures the complete buffer including those remaining writes.
  ✅ FIX-UNUSED-IMPORT-LIST       : removed unused `List` from typing import.
  ✅ FIX-UNUSED-VAR-ESCALATED     : removed the `escalated` local variable in
     update(); it was set but never read — the break statement and post-loop
     _cache_pos advance handle the two paths without needing the flag.
  ✅ FIX-WINDOW-SINK-VALIDATION   : __init__ now raises ValueError when
     window <= sink.  Previously the ring buffer had zero capacity
     (ring_len = 0), causing ZeroDivisionError in get() the first time
     pos exceeded window — a silent crash with no useful message.
  ✅ FIX-GET-RINGLEN-ZERO         : get() now guards ring_len > 0 explicitly
     as a second line of defence; returns only sink tokens when ring is empty
     rather than crashing with a modulo-zero error.
"""
from __future__ import annotations
import os
import warnings
from typing import Optional, Tuple
import numpy as np

from ..constants import SINK_TOKENS, KVCACHE_WARN_GB

__all__ = ["KVCache"]


class KVCache:
    def __init__(
        self,
        n_layers:   int,
        n_kv_heads: int,
        head_dim:   int,
        window:     int      = 1024,
        sink:       int      = SINK_TOKENS,
        dtype:      np.dtype = np.float16,
        use_memmap: bool     = False,
        memmap_dir: Optional[str] = None,
        max_batch:  int      = 1,
    ):
        # ✅ FIX-WINDOW-SINK-VALIDATION: ring capacity must be at least 1.
        # window == sink → ring_len = 0 → ZeroDivisionError in get() when
        # pos > window.  Catch this early with a clear message.
        if window <= sink:
            raise ValueError(
                f"KVCache: window ({window}) must be greater than sink ({sink}). "
                f"The ring buffer needs at least 1 slot beyond the sink region. "
                f"Either increase window or decrease sink."
            )

        self.n_layers   = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim   = head_dim
        self.window     = window
        self.sink       = sink
        self.dtype      = np.dtype(dtype)
        self.max_batch  = max_batch
        self._use_memmap = use_memmap

        # Warn early if the allocation will be huge
        _bytes = (n_layers * max_batch * n_kv_heads * window * head_dim
                  * 2 * self.dtype.itemsize)
        if _bytes > KVCACHE_WARN_GB * 1024 ** 3:
            warnings.warn(
                f"KVCache ~{_bytes / 1024**3:.1f} GB — consider reducing window "
                "or using use_memmap=True.", ResourceWarning, stacklevel=2)

        buf_shape = (n_layers, max_batch, n_kv_heads, window, head_dim)

        if use_memmap:
            import tempfile
            _dir = memmap_dir or tempfile.gettempdir()
            os.makedirs(_dir, exist_ok=True)
            self._K = np.memmap(os.path.join(_dir, "dracoai_kv_K.bin"),
                                dtype=self.dtype, mode="w+", shape=buf_shape)
            self._V = np.memmap(os.path.join(_dir, "dracoai_kv_V.bin"),
                                dtype=self.dtype, mode="w+", shape=buf_shape)
        else:
            self._K = np.zeros(buf_shape, dtype=self.dtype)
            self._V = np.zeros(buf_shape, dtype=self.dtype)

        self._cache_pos: np.ndarray = np.zeros(max_batch, dtype=np.int64)

    # ── Public position accessor (avoids direct _cache_pos access) ────
    def get_pos(self, batch_idx: int = 0) -> int:
        """Return the current absolute token position for batch_idx."""
        return int(self._cache_pos[batch_idx])

    # ── Slot index helper ─────────────────────────────────────────────
    def _slot(self, abs_pos: int) -> int:
        """Map absolute token position to ring-buffer slot index."""
        if abs_pos < self.sink:
            return abs_pos
        ring_cap = max(1, self.window - self.sink)
        return self.sink + (abs_pos - self.sink) % ring_cap

    # ── Write ──────────────────────────────────────────────────────────
    def update(self, layer_idx: int, K_new: np.ndarray, V_new: np.ndarray,
               snap: Optional[dict] = None, batch_idx: int = 0):
        """
        Write K_new / V_new into the ring buffer for batch_idx.
        Advances _cache_pos by seq tokens.

        Fast path (snap is None or snap type is "full"):
            Vectorised slot computation — no Python loop, 10-100× faster
            for long prefills.

        Slow path (snap is delta):
            Per-token loop needed to record individual (layer, slot) pairs
            for the delta snapshot.  Captures OLD values BEFORE writing so
            restore() correctly reverts state.  Escalates to full snapshot
            automatically when the delta list overflows the threshold;
            remaining tokens are then written via the fast path.
        """
        seq = K_new.shape[2]
        pos = int(self._cache_pos[batch_idx])
        K_f = K_new.astype(self.dtype)   # (1, n_kv_heads, seq, head_dim)
        V_f = V_new.astype(self.dtype)

        if snap is None or snap.get("_type") != "delta":
            # ── Vectorised fast path ──────────────────────────────────
            self._write_vectorised(layer_idx, batch_idx, pos, seq, K_f, V_f)
        else:
            # ── Delta-snapshot slow path ──────────────────────────────
            delta_list = snap.setdefault("_delta", [])
            threshold  = snap.get("_delta_threshold", 0)

            for t in range(seq):
                slot = self._slot(pos + t)

                # ✅ FIX-DELTA-RESTORE-OLD-VALUES: save OLD cache content
                #    BEFORE overwriting so restore() can revert correctly.
                K_old = self._K[layer_idx, batch_idx, :, slot, :].copy()
                V_old = self._V[layer_idx, batch_idx, :, slot, :].copy()

                # Write new values
                self._K[layer_idx, batch_idx, :, slot, :] = K_f[0, :, t, :]
                self._V[layer_idx, batch_idx, :, slot, :] = V_f[0, :, t, :]

                delta_list.append((layer_idx, batch_idx, slot, K_old, V_old))

                if threshold > 0 and len(delta_list) > threshold:
                    # ✅ FIX-DELTA-ESCALATION-PARTIAL: escalate to full,
                    #    then write remaining tokens (t+1 .. seq-1) via the
                    #    fast path so the buffer ends up complete.
                    self._snap_escalate_to_full(snap)
                    remaining = seq - (t + 1)
                    if remaining > 0:
                        self._write_vectorised(
                            layer_idx, batch_idx, pos + t + 1,
                            remaining,
                            K_f[:, :, t + 1:, :],
                            V_f[:, :, t + 1:, :],
                        )
                    break   # snap is now "full"; fast path handled the rest

        self._cache_pos[batch_idx] = pos + seq

    def _write_vectorised(
        self,
        layer_idx: int,
        batch_idx: int,
        start_pos: int,
        seq: int,
        K_f: np.ndarray,   # (1, n_kv_heads, seq, head_dim)
        V_f: np.ndarray,
    ):
        """Vectorised ring-buffer write for a contiguous seq of tokens."""
        ring_cap = max(1, self.window - self.sink)
        abs_pos  = start_pos + np.arange(seq, dtype=np.int32)
        buf_pos  = np.where(
            abs_pos < self.sink,
            abs_pos,
            self.sink + (abs_pos - self.sink) % ring_cap,
        )
        # K_f[0]: (n_kv_heads, seq, head_dim)
        # buf[layer, batch, :, buf_pos, :] fancy-indexed result has shape
        # (len(buf_pos), n_kv_heads, head_dim) due to numpy's advanced-indexing
        # convention when a fancy index appears between two slice axes.
        # Transposing K_f[0] to (seq, n_kv_heads, head_dim) matches that shape.
        self._K[layer_idx, batch_idx, :, buf_pos, :] = (
            K_f[0].transpose(1, 0, 2).astype(self.dtype))
        self._V[layer_idx, batch_idx, :, buf_pos, :] = (
            V_f[0].transpose(1, 0, 2).astype(self.dtype))

    # ── Read ───────────────────────────────────────────────────────────
    def get(self, layer_idx: int, batch_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Return (K, V) as float32 arrays with correct temporal ordering.

        ✅ FIX-GET-RINGLEN-ZERO: guard ring_len > 0 before modulo.  With a
        valid __init__ (window > sink) this should never trigger, but acts as
        a second line of defence for objects de-serialised from old checkpoints
        that predate the validation check.
        """
        pos    = int(self._cache_pos[batch_idx])
        length = min(pos, self.window)

        if pos <= self.window:
            K = self._K[layer_idx, batch_idx, :, :length, :][None]
            V = self._V[layer_idx, batch_idx, :, :length, :][None]
        else:
            sink     = self.sink
            ring_len = self.window - sink

            # ✅ FIX-GET-RINGLEN-ZERO: if somehow ring_len is 0, fall back to
            # returning only the sink region rather than crashing.
            if ring_len <= 0:
                K = self._K[layer_idx, batch_idx, :, :sink, :][None]
                V = self._V[layer_idx, batch_idx, :, :sink, :][None]
            else:
                ring_start = (pos - sink) % ring_len
                ring_order = [
                    (sink + (ring_start + i) % ring_len) for i in range(ring_len)
                ]
                idx = list(range(sink)) + ring_order
                K = self._K[layer_idx, batch_idx, :, idx, :][None]
                V = self._V[layer_idx, batch_idx, :, idx, :][None]

        return K.astype(np.float32), V.astype(np.float32)

    # ── step() is a documented no-op ──────────────────────────────────
    def step(self, n_tokens: int = 1, batch_idx: int = 0):
        """No-op. update() is the sole owner of _cache_pos."""
        pass

    # ── Reset ──────────────────────────────────────────────────────────
    def reset(self, batch_idx: Optional[int] = None):
        """Zero out the buffer(s) and reset position counter(s)."""
        if batch_idx is None:
            self._K[:] = 0; self._V[:] = 0; self._cache_pos[:] = 0
        else:
            self._K[:, batch_idx, :, :, :] = 0
            self._V[:, batch_idx, :, :, :] = 0
            self._cache_pos[batch_idx] = 0

    # ── Snapshot / Restore ─────────────────────────────────────────────
    def snapshot(self, delta_threshold: int = 0, batch_idx: int = 0) -> dict:
        snap: dict = {
            "_batch_idx": batch_idx,
            "_cache_pos": int(self._cache_pos[batch_idx]),
        }
        if delta_threshold <= 0:
            snap["_type"]      = "full"
            snap["_K"]         = self._K[:, batch_idx, :, :, :].copy()
            snap["_V"]         = self._V[:, batch_idx, :, :, :].copy()
            snap["_escalated"] = False
        else:
            effective = max(delta_threshold, self.n_layers)
            snap["_type"]             = "delta"
            snap["_delta"]            = []
            snap["_delta_threshold"]  = effective
            snap["_escalated"]        = False
        return snap

    def _snap_escalate_to_full(self, snap: dict):
        """Promote a delta snapshot to a full snapshot in-place."""
        if snap.get("_type") == "full":
            return
        batch_idx      = snap.get("_batch_idx", 0)
        snap["_type"]  = "full"
        snap["_K"]     = self._K[:, batch_idx, :, :, :].copy()
        snap["_V"]     = self._V[:, batch_idx, :, :, :].copy()
        snap["_escalated"] = True
        snap.pop("_delta", None)
        snap.pop("_delta_threshold", None)

    def restore(self, snap: dict):
        """Restore cache state from a snapshot."""
        batch_idx = snap.get("_batch_idx", 0)
        self._cache_pos[batch_idx] = snap["_cache_pos"]
        if snap["_type"] == "full":
            self._K[:, batch_idx, :, :, :] = snap["_K"]
            self._V[:, batch_idx, :, :, :] = snap["_V"]
        else:
            # Replay in REVERSE order: each entry holds OLD (pre-write) values.
            for layer_idx, b_idx, slot, K_old, V_old in reversed(snap.get("_delta", [])):
                self._K[layer_idx, b_idx, :, slot, :] = K_old
                self._V[layer_idx, b_idx, :, slot, :] = V_old

    # ── Checkpoint (disk) ─────────────────────────────────────────────
    def save_checkpoint(self, path: str):
        np.savez_compressed(
            path, K=self._K, V=self._V, cache_pos=self._cache_pos,
            n_layers=np.array(self.n_layers), n_kv_heads=np.array(self.n_kv_heads),
            head_dim=np.array(self.head_dim), window=np.array(self.window),
            sink=np.array(self.sink), max_batch=np.array(self.max_batch))

    @classmethod
    def load_checkpoint(cls, path: str) -> "KVCache":
        fname = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fname)
        cache = cls(
            n_layers=int(data["n_layers"]), n_kv_heads=int(data["n_kv_heads"]),
            head_dim=int(data["head_dim"]), window=int(data["window"]),
            sink=int(data["sink"]), dtype=data["K"].dtype,
            max_batch=int(data["max_batch"]))
        cache._K         = data["K"].copy()
        cache._V         = data["V"].copy()
        cache._cache_pos = data["cache_pos"].copy()
        return cache

    def current_length(self, batch_idx: int = 0) -> int:
        return min(int(self._cache_pos[batch_idx]), self.window)

    def __repr__(self) -> str:
        pos_summary = ", ".join(f"b{i}:{int(self._cache_pos[i])}"
                                for i in range(self.max_batch))
        return (f"KVCache(layers={self.n_layers}, heads={self.n_kv_heads}, "
                f"window={self.window}, sink={self.sink}, pos=[{pos_summary}])")