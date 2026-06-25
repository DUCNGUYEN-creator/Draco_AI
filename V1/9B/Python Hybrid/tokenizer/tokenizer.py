# DracoAI V1 — tokenizer/tokenizer.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
DracoAI V1 — BPETokenizer (Entry Point)
========================================
Qwen 3.5 9B Instruct compatible BPE Tokenizer with full multilingual,
emoji, streaming, and grapheme-aware decode support.

Architecture
------------
This class is a thin coordinator.  All heavy logic lives in sub-packages:

    unicode/        — normalisation, grapheme, emoji, script detection
    pretokenizer/   — language-aware text segmentation before BPE
    bpe/            — merge engine (O(N log N) heap-based)
    vocab/          — vocabulary, special tokens, extension vocab, serialisation
    streaming/      — grapheme-safe streaming & incremental decode
    chatml/         — ChatML template encoder/decoder

Quick start
-----------
    from tokenizer.tokenizer import BPETokenizer

    tok = BPETokenizer()
    ids  = tok.encode("Xin chao 👋 世界")
    text = tok.decode(ids)

    for chunk in tok.stream_decode(iter(ids)):
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterator, List, Optional, Tuple

# ── Sub-package imports ───────────────────────────────────────────────
from .config    import TokenizerConfig, DEFAULT_CONFIG
from .constants import (
    SPECIAL_TOKENS, SPECIAL_ID_TO_NAME, QWEN_BASE_END,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN,
)

from .unicode.normalize  import normalize as _unicode_normalize

from .pretokenizer.splitter import pretokenize

from .bpe.merges    import MergeEngine
from .vocab.vocab   import Vocabulary
from .vocab.special_tokens  import SpecialTokenRegistry
from .vocab.extension_vocab import ExtensionVocab
from .vocab         import serialization as _ser

from .streaming.decoder     import StreamDecoder
from .streaming.incremental import IncrementalDecoder

from .chatml.template import ChatMLTemplate


class BPETokenizer:
    """
    DracoAI BPE Tokenizer — multilingual, grapheme-aware, Qwen-compatible.

    Attributes
    ----------
    config : TokenizerConfig
        Runtime configuration (strict_utf8, grapheme_flush, …).
    """

    def __init__(self, config: Optional[TokenizerConfig] = None) -> None:
        self.config: TokenizerConfig = config or TokenizerConfig()
        self.config.validate()

        # ── Vocabulary ────────────────────────────────────────────────
        self._vocab        = Vocabulary()          # base bytes 0–255 + BPE merges
        self._special      = SpecialTokenRegistry( # special tokens + split regex
            extra=self.config.extra_special_tokens
        )
        self._ext_vocab    = ExtensionVocab()      # custom extension tokens

        # ── BPE engine ────────────────────────────────────────────────
        self._merge_engine = MergeEngine()

        # ── ChatML template ───────────────────────────────────────────
        self._chatml = ChatMLTemplate(
            encode_fn       = lambda t: self.encode(t, add_bos=False, add_eos=False),
            encode_bytes_fn = lambda b: self._bpe_encode_bytes(b),
        )

        # Newline cache (invalidated when merge table changes)
        self._nl_cache: Optional[List[int]] = None

    # ── Backward-compat properties ────────────────────────────────────

    @property
    def strict_utf8(self) -> bool:
        return self.config.strict_utf8

    @strict_utf8.setter
    def strict_utf8(self, value: bool) -> None:
        self.config.strict_utf8 = value

    @property
    def merges(self) -> Dict[Tuple[int, int], int]:
        """Read-only view of merge dict (legacy API compatibility)."""
        return self._merge_engine.merges

    @property
    def vocab(self) -> Dict[int, bytes]:
        """Read-only view of base vocabulary (legacy API compatibility)."""
        return self._vocab.id_to_bytes

    @property
    def inv_vocab(self) -> Dict[bytes, int]:
        return self._vocab.bytes_to_id

    # ── Internal helpers ──────────────────────────────────────────────

    def _utf8_errors(self) -> str:
        return "strict" if self.config.strict_utf8 else "replace"

    def _token_to_bytes(self, tid: int) -> bytes:
        """
        Convert a token ID to its byte representation.
        Priority: ext_vocab -> vocab -> special token name bytes.
        """
        b = self._ext_vocab.get_bytes(tid)
        if b is not None:
            return b
        b = self._vocab.get_bytes(tid)
        if b is not None:
            return b
        name = self._special.str_of(tid)
        if name is not None:
            return name.encode("utf-8")
        return b""

    def _is_valid_tid(self, tid: int) -> bool:
        """Return True if *tid* is a known token ID."""
        return (self._special.is_special(tid) or
                self._vocab.contains_id(tid) or
                self._ext_vocab.contains_id(tid))

    def _bpe_encode_bytes(self, raw: bytes) -> List[int]:
        """Encode raw bytes -> BPE token IDs."""
        byte_ids = list(raw)
        return self._merge_engine.encode(byte_ids)

    def _sanitize_text(self, text: str) -> str:
        """
        Apply security limits to input text before encoding:
        1. Truncate to config.max_input_length.
        2. Collapse excessive consecutive spaces (DoS protection).
        3. Strip excess combining marks (Zalgo text protection).
        """
        cfg = self.config

        # 1. Length truncation
        if len(text) > cfg.max_input_length:
            text = text[:cfg.max_input_length]

        # 2. Collapse excessive spaces
        if cfg.max_consecutive_spaces > 0:
            limit = cfg.max_consecutive_spaces
            text = re.sub(r' {' + str(limit + 1) + r',}', ' ' * limit, text)

        # 3. Strip excess combining marks (Zalgo protection)
        if cfg.max_combining_marks > 0:
            result_chars: List[str] = []
            run = 0
            for c in text:
                cat = unicodedata.category(c)
                if cat in ("Mn", "Mc", "Me"):
                    run += 1
                    if run <= cfg.max_combining_marks:
                        result_chars.append(c)
                else:
                    run = 0
                    result_chars.append(c)
            text = "".join(result_chars)

        return text

    # ── Encode ───────────────────────────────────────────────────────

    def encode(
        self,
        text:    str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
    ) -> List[int]:
        """
        Encode *text* to a list of token IDs.

        Steps:
        1. Security sanitisation.
        2. Unicode normalisation (configurable via config.normalisation_form).
        3. Split on special tokens (kept as-is).
        4. Pre-tokenise each segment (grapheme-safe, script-aware).
        5. BPE encode each pre-token.

        Parameters
        ----------
        text : str
            Input text.
        add_bos : Optional[bool]
            Prepend BOS token.  None (default) -> use config.add_bos_by_default.
        add_eos : Optional[bool]
            Append EOS token.  None (default) -> use config.add_eos_by_default.

        Returns
        -------
        List[int]
            Token IDs.
        """
        # Resolve defaults from config
        _add_bos = self.config.add_bos_by_default if add_bos is None else add_bos
        _add_eos = self.config.add_eos_by_default if add_eos is None else add_eos

        # Step 1: Security sanitisation (before normalisation to avoid length bypass)
        text = self._sanitize_text(text)

        # Step 2: Unicode normalisation
        text = _unicode_normalize(text, form=self.config.normalisation_form)

        result: List[int] = []
        if _add_bos:
            result.append(SPECIAL_TOKENS[BOS_TOKEN])

        # Step 3: Split on special tokens
        for part in self._special.split_pattern.split(text):
            if not part:
                continue
            if self._special.is_special_str(part):
                result.append(self._special.id_of(part))  # type: ignore[arg-type]
                continue

            for ext_part in self._ext_vocab.split(part):
                if not ext_part:
                    continue
                if self._ext_vocab.is_token_str(ext_part):
                    ext_id = self._ext_vocab.id_of(ext_part)
                    if ext_id is not None:
                        result.append(ext_id)
                    continue

                # Step 4 + 5: Pre-tokenise -> BPE
                for segment in pretokenize(
                    ext_part,
                    enable_cjk_split               = self.config.enable_cjk_split,
                    enable_thai_split              = self.config.enable_thai_split,
                    enable_scriptio_continua_split = self.config.enable_scriptio_continua_split,
                ):
                    byte_ids = list(segment.encode("utf-8"))
                    result.extend(self._merge_engine.encode(byte_ids))

        if _add_eos:
            result.append(SPECIAL_TOKENS[EOS_TOKEN])

        return result

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        system_prompt: Optional[str] = None,
    ) -> List[int]:
        """
        Encode a list of ChatML messages to token IDs.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Each dict: {"role": "user"|"assistant"|..., "content": "..."}.
        add_generation_prompt : bool
            Append the assistant turn opener (default True).
        system_prompt : Optional[str]
            Optional system message prepended before all others.

        Returns
        -------
        List[int]
            Token IDs for the full conversation.
        """
        return self._chatml.encode(
            messages,
            add_generation_prompt=add_generation_prompt,
            system_prompt=system_prompt,
        )

    def encode_with_context(
        self,
        text:        str,
        context_ids: List[int],
    ) -> List[int]:
        """
        Encode *text* and prepend *context_ids*.

        Parameters
        ----------
        text : str
            New text to encode.
        context_ids : List[int]
            Previously encoded token IDs.

        Returns
        -------
        List[int]
            context_ids + encode(text).
        """
        new_ids = self.encode(text, add_bos=False, add_eos=False)
        return context_ids + new_ids

    # ── Decode ───────────────────────────────────────────────────────

    def decode(
        self,
        ids:          List[int],
        skip_special: bool = True,
    ) -> str:
        """
        Decode a list of token IDs to text.

        Accumulates a single byte buffer across all IDs so multi-byte
        UTF-8 characters split across token boundaries are correctly
        reassembled.

        Parameters
        ----------
        ids : List[int]
            Token IDs.
        skip_special : bool
            If True (default), special tokens are omitted from output.

        Returns
        -------
        str
            Decoded text.
        """
        errors   = self._utf8_errors()
        byte_acc = b""

        for tid in ids:
            if not self._is_valid_tid(tid):
                continue
            if self._special.is_special(tid):
                if not skip_special:
                    name = self._special.str_of(tid) or ""
                    byte_acc += name.encode("utf-8")
                continue
            byte_acc += self._token_to_bytes(tid)

        return byte_acc.decode("utf-8", errors=errors)

    def decode_token(
        self,
        tid:          int,
        skip_special: bool = True,
    ) -> str:
        """
        Decode a single token ID to text.

        Note: For correct handling of multi-byte UTF-8 chars use
        stream_decode() or IncrementalDecoder instead.

        Parameters
        ----------
        tid : int
            A single token ID.
        skip_special : bool
            If True (default), returns "" for special tokens.

        Returns
        -------
        str
            Decoded text (may be empty for special or unknown tokens).
        """
        if not self._is_valid_tid(tid):
            return ""
        if self._special.is_special(tid):
            return "" if skip_special else (self._special.str_of(tid) or "")
        raw = self._token_to_bytes(tid)
        return raw.decode("utf-8", errors=self._utf8_errors())

    def stream_decode(
        self,
        ids:          Iterator[int],
        skip_special: bool = True,
    ) -> Iterator[str]:
        """
        Streaming decode: yields text chunks as tokens arrive.

        Flushes on grapheme cluster boundaries (when config.grapheme_flush
        is True) to prevent splitting emoji ZWJ sequences and Vietnamese
        diacritics.  Falls back to word-boundary flushing otherwise.

        Parameters
        ----------
        ids : Iterator[int]
            Iterable of token IDs.
        skip_special : bool
            If True (default), special tokens are not yielded.

        Yields
        ------
        str
            Text chunks (safe grapheme cluster boundaries).
        """
        decoder = StreamDecoder(
            token_to_bytes = self._token_to_bytes,
            is_special     = self._special.is_special,
            special_str    = self._special.str_of,
            errors         = self._utf8_errors(),
            grapheme_flush = self.config.grapheme_flush,
            skip_special   = skip_special,
        )
        yield from decoder.decode(ids)

    def make_incremental_decoder(
        self,
        skip_special: bool = True,
    ) -> IncrementalDecoder:
        """
        Create an IncrementalDecoder for real-time single-token decode.

        Returns
        -------
        IncrementalDecoder
            Call .push(token_id) for each token; .flush() at end of stream.
        """
        return IncrementalDecoder(
            token_to_bytes = self._token_to_bytes,
            is_special     = self._special.is_special,
            special_str    = self._special.str_of,
            errors         = self._utf8_errors(),
            skip_special   = skip_special,
        )

    # ── Vocabulary management ─────────────────────────────────────────

    def add_token(self, token_str: str) -> int:
        """
        Add a custom extension token.

        Idempotent: returns the same ID if the token was already added.

        Parameters
        ----------
        token_str : str
            Token string (e.g. "<draco_special>").

        Returns
        -------
        int
            Assigned token ID (>= QWEN_BASE_END).
        """
        return self._ext_vocab.add(token_str)

    def add_merge(self, pair: Tuple[int, int], new_id: int) -> None:
        """
        Add a BPE merge rule.

        Parameters
        ----------
        pair : Tuple[int, int]
            (left_token_id, right_token_id).
        new_id : int
            The merged token ID.
        """
        self._merge_engine.add_merge(pair, new_id)
        # Update vocab bytes for the merged token
        ba = self._token_to_bytes(pair[0])
        bb = self._token_to_bytes(pair[1])
        self._vocab.add(new_id, ba + bb)
        # Invalidate caches
        self._nl_cache = None
        self._chatml.invalidate_cache()

    def load_merges(self, merges: List[Tuple[Tuple[int, int], int]]) -> None:
        """
        Bulk-load BPE merge rules.

        Rules are applied in the order provided, which must match the
        training order (earlier merges first) so that vocab byte
        reconstruction for merged tokens is always available when needed.

        Parameters
        ----------
        merges : List[Tuple[Tuple[int, int], int]]
            List of ((left_id, right_id), merged_id).
        """
        self._merge_engine.load(merges)
        # Rebuild vocab bytes in rule order to ensure dependencies are met.
        # Each merged token's bytes depend on its constituent tokens, which must
        # already be in vocab by the time we process this rule.
        for (a, b), mid in merges:
            ba = self._token_to_bytes(a)
            bb = self._token_to_bytes(b)
            # If either side is still empty, it means the caller passed
            # out-of-order merges.  We still store whatever we have so
            # subsequent rules can at least partially resolve.
            self._vocab.add(mid, ba + bb)
        self._nl_cache = None
        self._chatml.invalidate_cache()

    def load_from_json(self, path: str) -> None:
        """Load tokenizer config from a Qwen-compatible tokenizer.json file."""
        _ser.load_from_hf_json(self, path)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save tokenizer to *path* directory."""
        _ser.save(self, path)

    @classmethod
    def load(cls, path: str, config: Optional[TokenizerConfig] = None) -> "BPETokenizer":
        """
        Load a tokenizer from a directory saved by .save().

        Parameters
        ----------
        path : str
            Directory containing ``tokenizer_draco.json``.
        config : Optional[TokenizerConfig]
            Override config (strict_utf8 etc.); checkpoint value used if None.
        """
        tok = cls(config=config)
        _ser.load(tok, path, load_config=config is None)
        return tok

    # ── Properties ────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (base + merges + extension + special)."""
        # Avoid constructing full copies of all dicts just for max().
        max_base  = max(self._vocab._id_to_bytes.keys(),  default=0)
        max_ext   = max(self._ext_vocab._id_to_bytes.keys(), default=0)
        max_spec  = max(self._special._id_to_str.keys(), default=0)
        return max(max_base, max_ext, max_spec) + 1
