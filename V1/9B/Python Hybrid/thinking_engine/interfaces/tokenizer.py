# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""thinking_engine.interfaces.tokenizer — contract for the BPE tokenizer."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        ...

    def decode(self, token_ids: List[int]) -> str:
        ...

    # Optional ChatML-aware alias — checked via hasattr() by callers,
    # not part of the structural Protocol contract (it's optional).
    # def encode_chat(self, messages: List[dict], add_generation_prompt: bool = True) -> List[int]: ...
