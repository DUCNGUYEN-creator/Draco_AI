# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
StreamingOutput
=================
New addition. Token-streaming wrapper: callers can register a callback
that receives each decoded token as it is generated, enabling real-time
display before the full response is assembled by ResponseBuilder.
"""

from __future__ import annotations

from typing import Callable, List, Optional


class StreamingOutput:
    def __init__(self, callback: Optional[Callable[[str], None]] = None) -> None:
        self._callback = callback
        self._tokens: List[str] = []

    def push_token(self, token_text: str) -> None:
        self._tokens.append(token_text)
        if self._callback is not None:
            try:
                self._callback(token_text)
            except Exception:
                pass

    def finalize(self) -> str:
        return "".join(self._tokens)
