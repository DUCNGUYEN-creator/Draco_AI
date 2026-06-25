# DracoAI V1 — tokenizer/streaming/incremental.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Incremental (Single-Token) Decoder
=======================================================
Decodes one token at a time for real-time UI scenarios (e.g. live
typing cursor, WebSocket streaming, token-level progress indicators).

Because a single token may contain only part of a UTF-8 codepoint,
the IncrementalDecoder maintains state between calls.  Call
``flush()`` when the stream ends to get any remaining bytes.

Usage
-----
.. code-block:: python

    dec = IncrementalDecoder(tokenizer)

    for token_id in stream:
        text = dec.push(token_id)
        if text:
            display(text)

    remainder = dec.flush()
    if remainder:
        display(remainder)
"""

from typing import Callable, Optional

from .buffers import ByteBuffer


class IncrementalDecoder:
    """
    Single-token incremental decoder with cross-token UTF-8 state.

    Parameters
    ----------
    token_to_bytes : Callable[[int], bytes]
        Maps a token ID to raw bytes.
    is_special : Callable[[int], bool]
        Returns True for special token IDs.
    special_str : Callable[[int], Optional[str]]
        Returns the string of a special token.
    errors : str
        "replace" or "strict".
    skip_special : bool
        If True (default), special tokens return "".
    """

    def __init__(
        self,
        token_to_bytes: Callable[[int], bytes],
        is_special:     Callable[[int], bool],
        special_str:    Callable[[int], Optional[str]],
        errors:         str  = "replace",
        skip_special:   bool = True,
    ) -> None:
        self._token_to_bytes = token_to_bytes
        self._is_special     = is_special
        self._special_str    = special_str
        self._skip_special   = skip_special
        self._buf            = ByteBuffer(errors=errors)

    def push(self, token_id: int) -> str:
        """
        Decode a single token and return any immediately available text.

        May return "" if the token's bytes form an incomplete UTF-8
        sequence (the bytes are buffered for the next call).

        Parameters
        ----------
        token_id : int
            A single token ID from the model output.

        Returns
        -------
        str
            Decoded text (possibly empty if bytes are still buffered).
        """
        if self._is_special(token_id):
            # Flush pending bytes first
            result = self._buf.flush()
            if self._skip_special:
                return result
            s = self._special_str(token_id) or ""
            return result + s

        raw = self._token_to_bytes(token_id)
        if not raw:
            return ""

        text, _ = self._buf.push(raw)
        return text

    def flush(self) -> str:
        """
        Flush any remaining buffered bytes at end-of-stream.

        Returns
        -------
        str
            Remaining decoded text (may contain replacement chars in
            "replace" mode if the stream ended mid-sequence).
        """
        return self._buf.flush()

    def reset(self) -> None:
        """Reset internal state for reuse with a new token stream."""
        self._buf.clear()
