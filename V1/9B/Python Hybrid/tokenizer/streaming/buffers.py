# DracoAI V1 — tokenizer/streaming/buffers.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer - Streaming Byte Buffers
==========================================
Manages partial UTF-8 byte accumulation across token boundaries.

The core problem: a single Unicode codepoint may be split across
multiple BPE tokens at the byte level.  We accumulate bytes until
Python's UTF-8 incremental decoder can emit complete text, while keeping
any incomplete tail buffered for the next token.
"""

import codecs
from typing import Tuple


class ByteBuffer:
    """
    Accumulates bytes across token boundaries and decodes complete
    UTF-8 sequences.

    ``errors="replace"`` emits U+FFFD for invalid byte sequences while
    still preserving incomplete trailing sequences until more bytes arrive.
    ``errors="strict"`` raises ``UnicodeDecodeError`` as soon as invalid
    bytes are observed.
    """

    def __init__(self, errors: str = "replace") -> None:
        self._errors: str = errors
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)
        self._buf: bytes = b""

    def push(self, data: bytes) -> Tuple[str, bytes]:
        """
        Add *data* to the buffer and decode as much as possible.

        Returns
        -------
        Tuple[str, bytes]
            ``(decoded_text, remaining_bytes)`` where ``remaining_bytes`` is
            the incomplete UTF-8 tail still held by the incremental decoder.
        """
        text = self._decoder.decode(data, final=False)
        self._buf = self._decoder.getstate()[0]
        return text, self._buf

    def _try_decode(self) -> Tuple[str, bytes]:
        """
        Decode any currently buffered bytes.

        Kept for compatibility with older internal callers; new code should
        call ``push``.
        """
        return self.push(b"")

    def flush(self) -> str:
        """
        Flush any remaining bytes using the configured error handler.

        Returns
        -------
        str
            Remaining decoded text. In ``replace`` mode this may contain
            U+FFFD if the stream ended mid-sequence.
        """
        result = self._decoder.decode(b"", final=True)
        self._decoder.reset()
        self._buf = b""
        return result

    def peek(self) -> bytes:
        """Return the current buffered bytes without consuming them."""
        self._buf = self._decoder.getstate()[0]
        return bytes(self._buf)

    def clear(self) -> None:
        """Discard any buffered bytes."""
        self._decoder.reset()
        self._buf = b""

    @property
    def has_pending(self) -> bool:
        """Return True if there are unprocessed bytes in the buffer."""
        self._buf = self._decoder.getstate()[0]
        return bool(self._buf)
