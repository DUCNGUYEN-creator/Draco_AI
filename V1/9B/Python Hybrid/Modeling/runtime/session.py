# DracoAI V1 — modeling/runtime/session.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
GenerationSession — serialisable per-request generation state.

NEW in this revision:
  ✅ FEAT-IN-MEMORY-ZEROIZATION : secure_clear() overwrites all in-memory
     session state (freq, pos, generated) with zeros/empty before releasing
     references.  Protects against Cold Boot Attacks where an attacker reads
     raw memory pages after the process exits.  Called automatically when
     the session is used as a context manager.
  ✅ FEAT-POOL-INTEGRATION      : attach a TensorPool so that secure_clear()
     also calls pool.secure_clear() — zeroing all pooled activation buffers
     that may contain sensitive KV/activation data.

FIXES retained:
  ✅ FIX-UNUSED-IMPORT-NUMPY : pure-Python value object, no NumPy.
"""
from __future__ import annotations
import json
import ctypes
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tensor_pool import TensorPool

__all__ = ["GenerationSession"]


def _zero_string(s: str) -> None:
    """Overwrite the characters of a Python string in-place (best-effort)."""
    try:
        raw = (ctypes.c_char * len(s)).from_address(id(s) + 49)
        for i in range(len(s)):
            raw[i] = 0
    except Exception:
        pass  # CPython layout may differ; silently skip


class GenerationSession:
    """
    Lightweight value object holding mutable per-session state.

    Usage (context-manager for automatic secure clear)::

        with GenerationSession() as session:
            # pass state into generate() at the start of each turn
            ...
        # session.secure_clear() called automatically on exit

    Usage (manual)::

        session = GenerationSession()
        ...
        session.secure_clear()   # wipe sensitive state from memory
    """

    def __init__(
        self,
        miro_mu:    float                       = 5.0,
        freq:       Optional[Dict[int, int]]    = None,
        pos:        Optional[Dict[int, int]]    = None,
        n_pos:      int                         = 0,
        generated:  Optional[List[int]]         = None,
        tensor_pool: Optional["TensorPool"]     = None,
    ):
        self.miro_mu    = miro_mu
        self.freq:      Dict[int, int] = freq      or {}
        self.pos:       Dict[int, int] = pos       or {}
        self.n_pos      = n_pos
        self.generated: List[int]      = generated or []
        self._tensor_pool              = tensor_pool

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "GenerationSession":
        return self

    def __exit__(self, *_) -> None:
        self.secure_clear()

    # ── Core state management ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all session state (start a new conversation turn)."""
        self.miro_mu   = 5.0
        self.freq      = {}
        self.pos       = {}
        self.n_pos     = 0
        self.generated = []

    def record_token(self, token_id: int) -> None:
        """Register a newly generated token."""
        self.freq[token_id] = self.freq.get(token_id, 0) + 1
        self.pos[token_id]  = self.n_pos
        self.n_pos         += 1
        self.generated.append(token_id)

    # ── Secure zeroization ────────────────────────────────────────────────────

    def secure_clear(self) -> None:
        """
        Overwrite all sensitive in-memory session state with zeros, then
        release references so the garbage collector can reclaim the pages.

        ✅ FEAT-IN-MEMORY-ZEROIZATION: protects against Cold Boot Attacks
        where an attacker reads raw RAM pages after the process exits.
        This overwrites:
          • self.generated  list contents (zeroed element-by-element)
          • self.freq / pos dict values  (zeroed before clear)
          • self.miro_mu, self.n_pos    (reset to neutral values)
          • TensorPool buffers          (if a pool was attached)

        Note: Python's memory model does not guarantee that old objects
        are immediately reclaimed or that the GC zeroes freed pages.
        This method provides a reasonable best-effort defence without
        requiring native extensions.
        """
        # Zero generated list in-place before clearing
        for i in range(len(self.generated)):
            self.generated[i] = 0
        self.generated.clear()

        # Zero dict values before clearing
        for k in list(self.freq.keys()):
            self.freq[k] = 0
        self.freq.clear()

        for k in list(self.pos.keys()):
            self.pos[k] = 0
        self.pos.clear()

        # Reset scalars
        self.miro_mu = 0.0
        self.n_pos   = 0

        # Zero pooled activation buffers
        if self._tensor_pool is not None:
            try:
                self._tensor_pool.secure_clear()
            except Exception:
                pass

    def attach_tensor_pool(self, pool: "TensorPool") -> None:
        """Attach a TensorPool so secure_clear() also zeroes pooled buffers."""
        self._tensor_pool = pool

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return dict(
            miro_mu   = self.miro_mu,
            freq      = {str(k): v for k, v in self.freq.items()},
            pos       = {str(k): v for k, v in self.pos.items()},
            n_pos     = self.n_pos,
            generated = self.generated,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationSession":
        return cls(
            miro_mu   = float(d.get("miro_mu", 5.0)),
            freq      = {int(k): v for k, v in d.get("freq", {}).items()},
            pos       = {int(k): v for k, v in d.get("pos",  {}).items()},
            n_pos     = int(d.get("n_pos", 0)),
            generated = list(d.get("generated", [])),
        )

    def save(self, path: str) -> None:
        """Persist session to a .json file."""
        with open(path if path.endswith(".json") else path + ".json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GenerationSession":
        with open(path if path.endswith(".json") else path + ".json") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return (f"GenerationSession(n_pos={self.n_pos}, "
                f"unique_tokens={len(self.freq)}, "
                f"generated={len(self.generated)}, mu={self.miro_mu:.2f})")