# DracoAI V1 — modeling/runtime/session.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
GenerationSession — serialisable per-request generation state.

Captures mutable state that must survive across generate() calls
(e.g., multi-turn conversations, checkpoint/resume).

State held:
  - miro_mu      : Mirostat target complexity (float)
  - freq         : per-token frequency counts (dict)
  - pos          : last seen position of each token (dict)
  - n_pos        : total token positions processed (int)
  - generated    : list of generated token IDs so far
"""
from __future__ import annotations
import json
from typing import Dict, List, Optional
import numpy as np

__all__ = ["GenerationSession"]


class GenerationSession:
    """
    Lightweight value object holding mutable per-session state.

    Usage::

        session = GenerationSession()
        # pass state into generate() at the start of each turn
    """

    def __init__(
        self,
        miro_mu:   float               = 5.0,
        freq:      Optional[Dict[int, int]] = None,
        pos:       Optional[Dict[int, int]] = None,
        n_pos:     int                  = 0,
        generated: Optional[List[int]] = None,
    ):
        self.miro_mu    = miro_mu
        self.freq:      Dict[int, int] = freq      or {}
        self.pos:       Dict[int, int] = pos       or {}
        self.n_pos      = n_pos
        self.generated: List[int]      = generated or []

    def reset(self):
        """Clear all session state (start a new conversation turn)."""
        self.miro_mu   = 5.0
        self.freq      = {}
        self.pos       = {}
        self.n_pos     = 0
        self.generated = []

    def record_token(self, token_id: int):
        """Register a newly generated token."""
        self.freq[token_id] = self.freq.get(token_id, 0) + 1
        self.pos[token_id]  = self.n_pos
        self.n_pos         += 1
        self.generated.append(token_id)

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

    def save(self, path: str):
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