# DracoAI V1 — tokenizer/streaming/decoder.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI Tokenizer — Streaming Decoder
=======================================
Grapheme-aware streaming decode of token ID sequences.

Key invariants:
- A single persistent ByteBuffer handles cross-token UTF-8 reassembly.
- Text is only yielded at complete grapheme cluster boundaries
  (when grapheme_flush=True) to prevent splitting emoji ZWJ sequences
  and Vietnamese diacritics.
- On a special token or end-of-stream, any pending bytes are flushed
  with error handling before the special token itself is emitted.
"""

from typing import Callable, Iterator, Optional

from .buffers import ByteBuffer
from ..constants import FLUSH_CHARS
from ..unicode.grapheme import is_safe_to_yield, safe_flush_point


class StreamDecoder:
    """
    Streaming decoder for token ID sequences.

    Parameters
    ----------
    token_to_bytes : Callable[[int], bytes]
        Maps a token ID to its raw bytes (b"" for unknown IDs).
    is_special : Callable[[int], bool]
        Returns True for special token IDs.
    special_str : Callable[[int], Optional[str]]
        Returns the string representation of a special token ID.
    errors : str
        UTF-8 decode error mode: "replace" (default) or "strict".
    grapheme_flush : bool
        If True (default), flush on grapheme cluster boundaries for
        safe emoji/Vietnamese streaming.  If False, flush on any
        character in FLUSH_CHARS.
    skip_special : bool
        If True (default), special tokens are not yielded.
    """

    def __init__(
        self,
        token_to_bytes:  Callable[[int], bytes],
        is_special:      Callable[[int], bool],
        special_str:     Callable[[int], Optional[str]],
        errors:          str  = "replace",
        grapheme_flush:  bool = True,
        skip_special:    bool = True,
    ) -> None:
        self._token_to_bytes  = token_to_bytes
        self._is_special      = is_special
        self._special_str     = special_str
        self._errors          = errors
        self._grapheme_flush  = grapheme_flush
        self._skip_special    = skip_special

    def _should_flush(self, text: str) -> bool:
        """Return True if *text* ends at a safe yield boundary."""
        if not text:
            return False

        if self._grapheme_flush:
            # Flush only when the text ends on a complete grapheme cluster
            # AND the last grapheme is a word-boundary character.
            if not is_safe_to_yield(text):
                return False
            return text[-1] in FLUSH_CHARS
        else:
            # Legacy: flush on any flush-char character.
            return text[-1] in FLUSH_CHARS

    def _flush_text_buf(self, text_buf: str) -> Iterator[str]:
        """
        Yield the safe prefix of *text_buf* and return the remainder.

        Yields nothing if nothing is safe to emit yet.
        This is an internal helper yielded inline; callers update text_buf
        themselves after calling it.
        """
        if not text_buf:
            return
        if self._grapheme_flush:
            safe = safe_flush_point(text_buf)
            if safe > 0:
                yield text_buf[:safe]
        else:
            yield text_buf

    def decode(self, ids: Iterator[int]) -> Iterator[str]:
        """
        Streaming decode token IDs -> text chunks.

        Yields
        ------
        str
            Text chunks at word/grapheme boundaries.  The final chunk
            is flushed at end-of-stream regardless of boundary.
        """
        byte_buf  = ByteBuffer(errors=self._errors)
        text_buf  = ""

        for tid in ids:
            if self._is_special(tid):
                # ── Flush all pending bytes and text before the special token ──
                # Step 1: drain any bytes that are now decodable.
                decoded, _ = byte_buf.push(b"")
                text_buf  += decoded

                # Step 2: yield safe prefix of text_buf.
                if text_buf:
                    if self._grapheme_flush:
                        safe = safe_flush_point(text_buf)
                        if safe > 0:
                            yield text_buf[:safe]
                            text_buf = text_buf[safe:]
                    else:
                        yield text_buf
                        text_buf = ""

                # Step 3: yield any remaining text (unsafe grapheme tail).
                if text_buf:
                    yield text_buf
                    text_buf = ""

                # Step 4: flush incomplete UTF-8 bytes via error handler.
                if byte_buf.has_pending:
                    tail = byte_buf.flush()
                    if tail:
                        yield tail

                # Step 5: emit the special token string if not skipped.
                if not self._skip_special:
                    s = self._special_str(tid)
                    if s:
                        yield s
                continue

            raw = self._token_to_bytes(tid)
            if not raw:
                continue

            chunk, _ = byte_buf.push(raw)
            text_buf += chunk

            # Flush on safe boundary.
            if self._should_flush(text_buf):
                if self._grapheme_flush:
                    safe = safe_flush_point(text_buf)
                    if safe > 0:
                        yield text_buf[:safe]
                        text_buf = text_buf[safe:]
                else:
                    yield text_buf
                    text_buf = ""

        # ── Final flush: remaining bytes + text ──
        if byte_buf.has_pending:
            text_buf += byte_buf.flush()

        if text_buf:
            yield text_buf
