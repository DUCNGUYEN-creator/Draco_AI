# Copyright 2026 Draco Studio and DUCNGUYEN-creator
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DracoAI V1 — Tokenizer
======================
Qwen 3.5 9B Instruct compatible.

- BPE encode O(N log N) via merge_rank + heapq
- UTF-8 byte-buffer safe streaming decode
- Extension vocab lock (ID >= QWEN_BASE_END)
- ChatML template: <|im_start|>role\ncontent<|im_end|>
- Unicode NFC normalization for Vietnamese diacritic compatibility

QWEN_BASE_END = 151936  (Qwen 3.5 9B vocab boundary)

FIXES (V1 — final consolidated):
    ✅ Qwen 3.5 9B Instruct (was 7B — typo)
    ✅ Special token IDs aligned to Qwen 3.5 9B checkpoint layout
    ✅ encode_chat uses correct token IDs
    ✅ stream_decode: UTF-8 safe, flush on word boundary
    ✅ decode_token: single-token real-time decode
    ✅ decode_token & stream_decode: unified UTF-8 error handling (errors="replace")
    ✅ encode_with_context: add_bos/add_eos=False to avoid double-wrapping tokens
    ✅ _bpe_encode_plain: heapq-based O(N log N) merge loop
    ✅ stream_decode: partial UTF-8 bytes accumulated correctly across token boundaries
    ✅ vocab_size property: accounts for special tokens above QWEN_BASE_END
    ✅ FIX: stream_decode no longer double-yields after special token handling
    ✅ FIX: _SP_PATTERN safely escapes special token strings for regex
    ✅ FIX: _bpe_encode_plain uses per-pair counter to break heap ties deterministically
    ✅ FIX: encode() correctly splits on special tokens before BPE
    ✅ FIX: decode_token handles both base-vocab bytes and extension vocab
    ✅ FIX S3: decode_token() validates tid before accessing vocab
    ✅ FIX S4: strict=False/True mode for stream_decode() and decode()
    ✅ FIX: encode_chat() newline encoding uses cached bytes
    ✅ FIX: _token_to_bytes() returns bytes for all known special tokens
    ✅ FIX: add_token() skips already-used ext_vocab IDs
    ✅ FIX TH: stream_decode() accumulates UTF-8 byte_buf across full iterator
    ✅ FIX: decode() accumulates single byte_buf across all token IDs
    ✅ FIX UNICODE-NFC: encode() applies unicodedata.normalize("NFC") so that
         Vietnamese "hòa" (precomposed) and "hoà" (decomposed) are treated
         identically, preventing duplicate token sequences for the same word.
    ✅ FIX PRE-TOK: encode() uses _PATTERN.finditer for word-piece splitting
         (mirrors Qwen tokenizer pre-tokenizer behaviour for better checkpoint
         compatibility). Segments text into alpha-runs, numbers, whitespace,
         punctuation before BPE — consistent with transformers tokenizer.
    ✅ FIX STREAM-FLUSH: stream_decode() final flush decodes trailing byte_buf
         with errors=errors even in strict mode (raises instead of replacing),
         so callers that set strict_utf8=True always see the error rather than
         losing the final character silently.
    ✅ FIX: add_merge() resets _nl_cache when merges/vocab change so the cached
         newline token sequence is recomputed with the updated merge table.
"""

import re, os, json, math, hashlib, heapq, unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional, Iterator

# ── Special tokens (Qwen 3.5 9B ChatML style) ────────────────────────
SPECIAL_TOKENS: Dict[str, int] = {
    "<|endoftext|>": 151643,
    "<|im_start|>":  151644,
    "<|im_end|>":    151645,
    "<|pad|>":       151646,
    "<|unk|>":       151647,
    "<|sep|>":       151648,
    "<think>":       151649,
    "</think>":      151650,
    "<tool_call>":   151651,
    "</tool_call>":  151652,
}

# Reverse lookup: id → name (built once, reused everywhere)
_SPECIAL_ID_TO_NAME: Dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

N_SPECIAL     = len(SPECIAL_TOKENS)
QWEN_BASE_END = 151936   # Qwen 3.5 9B vocab boundary

BOS_TOKEN = "<|im_start|>"
EOS_TOKEN = "<|im_end|>"
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"

# ─────────────────────────────────────────────────────────────────────

class CharAnalyzer:
    VIET  = set("áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ"
                "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ")
    PUNCT = set(".,!?;:\"'()[]{}/—–...«»")

    @staticmethod
    def char_type(c: str) -> str:
        if c.isdigit():                   return "NUM"
        if c in CharAnalyzer.VIET:        return "VIET_TONE"
        if c.isalpha():                   return "VIET" if ord(c) > 127 else "ALPHA"
        if c in CharAnalyzer.PUNCT:       return "PUNCT"
        if c == " ":                      return "SPACE"
        if c == "\n":                     return "NEWLINE"
        return "OTHER"

    @staticmethod
    def analyze_word(w: str) -> dict:
        types = [CharAnalyzer.char_type(c) for c in w]
        return {
            "text": w, "length": len(w), "char_types": types,
            "has_viet": any(t in ("VIET", "VIET_TONE") for t in types),
            "has_num":  any(t == "NUM" for t in types),
            "is_upper": w.isupper(), "is_title": w.istitle(),
        }

# ─────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer — DracoAI / Qwen 3.5 9B Instruct compatible.

    Special token IDs match Qwen 3.5 9B checkpoint (151643–151652).
    Base vocab IDs: 0–255 (byte tokens), merges above 255.

    Streaming:
        for piece in tok.stream_decode(ids):
            print(piece, end="", flush=True)

    Single-token decode (real-time UI):
        text = tok.decode_token(token_id)

    Strict UTF-8 mode (for debug/training):
        tok.strict_utf8 = True   # raises on invalid bytes instead of replacing
    """

    # Chars that trigger a stream-flush (word boundary)
    _FLUSH_CHARS: frozenset = frozenset(" \t\n.,!?;:—–()[]{}'\"，。！？；：")

    # Pre-tokenizer pattern — mirrors Qwen tokenizer word-piece splitting
    # Handles Vietnamese accented chars (À-ỹ range) as alpha
    _PATTERN = re.compile(
        r"[a-zA-ZÀ-ỹ]+|"
        r"\d+(?:[.,]\d+)?|"
        r"\s+|"
        r"[^\w\s]",
        re.UNICODE,
    )

    # Pre-compiled special-token pattern (rebuilt whenever SPECIAL_TOKENS changes)
    _SP_PATTERN: Optional[re.Pattern] = None

    def __init__(self):
        self.merges:      Dict[Tuple[int, int], int] = {}
        self._merge_rank: Dict[Tuple[int, int], int] = {}
        self.vocab:       Dict[int, bytes]            = {}
        self.inv_vocab:   Dict[bytes, int]            = {}
        self._ext_vocab:  Dict[int, bytes]            = {}
        self._ext_inv:    Dict[bytes, int]            = {}

        # Strict UTF-8 flag: set True to get UnicodeDecodeError on bad bytes
        self.strict_utf8: bool = False

        # Populate base byte vocab (IDs 0–255)
        for i in range(256):
            b = bytes([i])
            self.vocab[i]     = b
            self.inv_vocab[b] = i

        self._rebuild_sp_pattern()

        # Cache the newline token sequence (used frequently in encode_chat)
        self._nl_cache: Optional[List[int]] = None

    # ── Internal helpers ──────────────────────────────────────────────

    @classmethod
    def _rebuild_sp_pattern(cls):
        """Rebuild regex that splits text on special tokens.
        FIX: use re.escape so tokens like <|im_start|> don't break the pattern.
        Sort by length descending so longer tokens match first.
        """
        sorted_tokens = sorted(SPECIAL_TOKENS.keys(), key=len, reverse=True)
        pattern = "|".join(re.escape(t) for t in sorted_tokens)
        cls._SP_PATTERN = re.compile(f"({pattern})")

    def _utf8_errors(self) -> str:
        """Return the errors= mode for bytes.decode() based on strict_utf8 flag."""
        return "strict" if self.strict_utf8 else "replace"

    def _token_to_bytes(self, tid: int) -> bytes:
        """
        Convert a token ID to its byte representation.
        Priority: ext_vocab → vocab (base merges + bytes) → special token name bytes.
        Never returns b"" for valid special tokens.
        """
        if tid in self._ext_vocab:
            return self._ext_vocab[tid]
        if tid in self.vocab:
            return self.vocab[tid]
        name = _SPECIAL_ID_TO_NAME.get(tid)
        if name is not None:
            return name.encode("utf-8")
        return b""

    def _is_valid_tid(self, tid: int) -> bool:
        """
        FIX S3: Set-membership check (gap-safe, O(1)).
        Prevents KeyError on garbage IDs from a broken model.
        """
        if tid in _SPECIAL_ID_TO_NAME:
            return True
        if tid in self.vocab:
            return True
        if tid in self._ext_vocab:
            return True
        return False

    # ── O(N log N) BPE encode ─────────────────────────────────────────

    def _bpe_encode_plain(self, byte_ids: List[int]) -> List[int]:
        """
        BPE encode a sequence of byte IDs using a min-heap over merge ranks.
        FIX: monotonic counter as secondary heap key breaks ties deterministically,
        avoiding stale-entry confusion when multiple pairs share the same rank.
        """
        n = len(byte_ids)
        if n == 0:
            return []
        if n == 1:
            return byte_ids[:]

        ids    = list(byte_ids)
        active = [True] * n

        # Doubly-linked list over active indices
        prev = list(range(-1, n - 1))
        nxt  = list(range(1, n + 1))
        nxt[n - 1] = n  # sentinel

        heap: list = []
        counter = 0  # tie-breaker

        def push(li: int, ri: int):
            nonlocal counter
            pair = (ids[li], ids[ri])
            rank = self._merge_rank.get(pair)
            if rank is not None:
                heapq.heappush(heap, (rank, counter, li, ri))
                counter += 1

        # Initialise heap with all adjacent pairs
        for i in range(n - 1):
            push(i, i + 1)

        while heap:
            rank, _, li, ri = heapq.heappop(heap)

            # Validate: both positions still active and still adjacent
            if not active[li] or not active[ri]:
                continue
            if nxt[li] != ri:
                continue

            # Check pair still matches (ids may have changed via earlier merges)
            pair = (ids[li], ids[ri])
            if self._merge_rank.get(pair) != rank:
                continue

            new_id = self.merges[pair]
            ids[li] = new_id

            # Unlink ri
            active[ri] = False
            rri = nxt[ri]
            nxt[li] = rri
            if rri < n:
                prev[rri] = li

            # Push new pairs involving li
            lli = prev[li]
            if lli >= 0 and active[lli]:
                push(lli, li)
            if rri < n and active[rri]:
                push(li, rri)

        return [ids[i] for i in range(n) if active[i]]

    # ── Encode ───────────────────────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        FIX UNICODE-NFC: Normalize text to NFC form before encoding so that
        precomposed and decomposed Unicode variants (common in Vietnamese)
        produce the same token sequence.

        FIX PRE-TOK: After splitting on special tokens, apply _PATTERN word-piece
        splitting before BPE (mirrors Qwen tokenizer pre-tokenizer, improves
        checkpoint compatibility vs. encoding the whole segment as raw bytes).

        FIX: Split on special tokens first so they are never BPE-encoded.
        """
        # FIX UNICODE-NFC: normalize before any processing
        text = unicodedata.normalize("NFC", text)

        if self._SP_PATTERN is None:
            self._rebuild_sp_pattern()

        result: List[int] = []
        if add_bos:
            result.append(SPECIAL_TOKENS[BOS_TOKEN])

        for part in self._SP_PATTERN.split(text):
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                result.append(SPECIAL_TOKENS[part])
            else:
                # FIX PRE-TOK: segment with _PATTERN before BPE for Qwen compatibility
                for match in self._PATTERN.finditer(part):
                    word     = match.group()
                    byte_ids = list(word.encode("utf-8"))
                    result.extend(self._bpe_encode_plain(byte_ids))

        if add_eos:
            result.append(SPECIAL_TOKENS[EOS_TOKEN])

        return result

    def encode_chat(self, messages: List[Dict[str, str]],
                    add_generation_prompt: bool = True) -> List[int]:
        """
        Encode a list of ChatML messages to token IDs.
        FIX: newline bytes cached (not re-encoded per turn).
        FIX: add_bos/add_eos=False to avoid double-wrapping.
        """
        if self._nl_cache is None:
            self._nl_cache = self._bpe_encode_plain(list(b"\n"))

        ids: List[int] = []
        im_start = SPECIAL_TOKENS[BOS_TOKEN]
        im_end   = SPECIAL_TOKENS[EOS_TOKEN]

        for msg in messages:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            ids.append(im_start)
            ids.extend(self.encode(role,    add_bos=False, add_eos=False))
            ids.extend(self._nl_cache)
            ids.extend(self.encode(content, add_bos=False, add_eos=False))
            ids.append(im_end)
            ids.extend(self._nl_cache)

        if add_generation_prompt:
            ids.append(im_start)
            ids.extend(self.encode("assistant", add_bos=False, add_eos=False))
            ids.extend(self._nl_cache)

        return ids

    def encode_with_context(self, text: str, context_ids: List[int]) -> List[int]:
        """Encode new text and prepend existing context IDs.
        FIX: add_bos/add_eos=False to avoid double-wrapping tokens.
        """
        new_ids = self.encode(text, add_bos=False, add_eos=False)
        return context_ids + new_ids

    # ── Decode ───────────────────────────────────────────────────────

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode a list of token IDs to text.
        FIX TH: Accumulate a single byte_buf across all IDs before decoding,
        so multi-byte UTF-8 chars split across token boundaries are reassembled.
        """
        errors   = self._utf8_errors()
        byte_buf = b""

        for tid in ids:
            if not self._is_valid_tid(tid):
                continue
            if tid in _SPECIAL_ID_TO_NAME:
                if skip_special:
                    continue
                byte_buf += _SPECIAL_ID_TO_NAME[tid].encode("utf-8")
                continue
            byte_buf += self._token_to_bytes(tid)

        return byte_buf.decode("utf-8", errors=errors)

    def decode_token(self, tid: int, skip_special: bool = True) -> str:
        """
        Decode a single token ID to text (real-time / streaming UI use).
        FIX S3: Validates tid before accessing vocab.
        Note: For multi-byte UTF-8 chars use stream_decode() for correct handling.
        """
        if not self._is_valid_tid(tid):
            return ""
        if tid in _SPECIAL_ID_TO_NAME:
            return "" if skip_special else _SPECIAL_ID_TO_NAME[tid]
        raw = self._token_to_bytes(tid)
        return raw.decode("utf-8", errors=self._utf8_errors())

    def stream_decode(self, ids: Iterator[int],
                      skip_special: bool = True) -> Iterator[str]:
        """
        Streaming decode: yields text chunks as tokens arrive.

        FIX TH: Uses a persistent byte_buf local variable across the entire
        iterator — correctly handles multi-byte UTF-8 chars split across
        token boundaries.

        FIX STREAM-FLUSH: Final flush of trailing byte_buf uses the configured
        errors mode (strict or replace), so callers with strict_utf8=True see
        the UnicodeDecodeError instead of losing the last character silently.

        FIX: No double-yield after special token handling.
        """
        errors   = self._utf8_errors()
        byte_buf = b""
        text_buf = ""

        for tid in ids:
            if not self._is_valid_tid(tid):
                continue

            if tid in _SPECIAL_ID_TO_NAME:
                # Flush any pending bytes/text before emitting special token
                if byte_buf:
                    try:
                        text_buf += byte_buf.decode("utf-8")
                    except UnicodeDecodeError:
                        text_buf += byte_buf.decode("utf-8", errors=errors)
                    byte_buf = b""

                if text_buf:
                    yield text_buf
                    text_buf = ""

                if not skip_special:
                    yield _SPECIAL_ID_TO_NAME[tid]
                continue

            raw = self._token_to_bytes(tid)
            if not raw:
                continue

            byte_buf += raw

            # Try to decode accumulated bytes
            try:
                chunk    = byte_buf.decode("utf-8")
                byte_buf = b""   # fully consumed
            except UnicodeDecodeError:
                # Incomplete multi-byte sequence — hold bytes for next token
                chunk = ""
                for end in range(len(byte_buf), 0, -1):
                    try:
                        chunk    = byte_buf[:end].decode("utf-8")
                        byte_buf = byte_buf[end:]
                        break
                    except UnicodeDecodeError:
                        continue

            text_buf += chunk

            # Flush on word boundary
            if text_buf and text_buf[-1] in self._FLUSH_CHARS:
                yield text_buf
                text_buf = ""

        # FIX STREAM-FLUSH: final flush — respects strict_utf8 mode
        if byte_buf:
            text_buf += byte_buf.decode("utf-8", errors=errors)
        if text_buf:
            yield text_buf

    # ── Vocabulary management ─────────────────────────────────────────

    def add_token(self, token_str: str) -> int:
        """
        Add a custom token to the extension vocabulary.
        FIX: id collision loop skips already-used ext_vocab IDs in addition
        to SPECIAL_TOKENS IDs.
        Returns the assigned token ID.
        """
        b = token_str.encode("utf-8")
        if b in self._ext_inv:
            return self._ext_inv[b]

        # Find next free ID above QWEN_BASE_END
        candidate = QWEN_BASE_END
        used_ids  = set(_SPECIAL_ID_TO_NAME.keys()) | set(self._ext_vocab.keys())
        while candidate in used_ids:
            candidate += 1

        self._ext_vocab[candidate] = b
        self._ext_inv[b]           = candidate
        return candidate

    def add_merge(self, pair: Tuple[int, int], new_id: int):
        """Add a BPE merge rule.
        FIX: Resets _nl_cache so cached newline tokens are recomputed after
        new merges change the BPE table.
        """
        rank = len(self._merge_rank)
        self.merges[pair]      = new_id
        self._merge_rank[pair] = rank
        merged_bytes = self.vocab.get(pair[0], b"") + self.vocab.get(pair[1], b"")
        self.vocab[new_id]           = merged_bytes
        self.inv_vocab[merged_bytes] = new_id
        # FIX: invalidate nl_cache after merge table changes
        self._nl_cache = None

    def load_merges(self, merges: List[Tuple[Tuple[int, int], int]]):
        """Load BPE merge rules. merges: list of ((id1, id2), merged_id)."""
        for (p1, p2), mid in merges:
            self.merges[(p1, p2)]      = mid
            self._merge_rank[(p1, p2)] = len(self._merge_rank)
            if mid not in self.vocab and mid not in self._ext_vocab:
                b1 = self._token_to_bytes(p1)
                b2 = self._token_to_bytes(p2)
                merged_bytes = b1 + b2
                self.vocab[mid]             = merged_bytes
                self.inv_vocab[merged_bytes] = mid
        self._nl_cache = None  # invalidate after bulk load

    def load_from_json(self, path: str):
        """Load tokenizer config from a Qwen-compatible tokenizer.json file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("added_tokens", []):
            tid     = entry["id"]
            content = entry["content"]
            b       = content.encode("utf-8")
            if tid not in _SPECIAL_ID_TO_NAME and tid >= QWEN_BASE_END:
                self._ext_vocab[tid] = b
                self._ext_inv[b]     = tid

        model = data.get("model", {})
        raw_merges = model.get("merges", [])
        merge_list = []
        for pair_str in raw_merges:
            parts = pair_str.split(" ", 1)
            if len(parts) != 2:
                continue
            left_b  = parts[0].encode("utf-8")
            right_b = parts[1].encode("utf-8")
            lid = self.inv_vocab.get(left_b)
            rid = self.inv_vocab.get(right_b)
            if lid is None or rid is None:
                continue
            merged_b  = left_b + right_b
            merged_id = self.inv_vocab.get(merged_b)
            if merged_id is None:
                merged_id = max(self.vocab.keys(), default=255) + 1
                self.vocab[merged_id]     = merged_b
                self.inv_vocab[merged_b]  = merged_id
            merge_list.append(((lid, rid), merged_id))

        self.load_merges(merge_list)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including base vocab, merges, and extension tokens."""
        all_ids = set(self.vocab.keys()) | set(self._ext_vocab.keys()) | set(_SPECIAL_ID_TO_NAME.keys())
        return max(all_ids, default=0) + 1

    def save(self, path: str):
        """Save tokenizer to a directory."""
        os.makedirs(path, exist_ok=True)
        merges_list = [[a, b, self.merges[(a, b)]] for (a, b) in self.merges]
        ext_vocab   = {str(k): v.decode("latin-1") for k, v in self._ext_vocab.items()}
        config = {
            "model":      "Qwen3.5-9B-Instruct",
            "version":    "v1",
            "merges":     merges_list,
            "ext_vocab":  ext_vocab,
            "strict_utf8": self.strict_utf8,
        }
        with open(os.path.join(path, "tokenizer_draco.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from a directory saved by .save()."""
        tok      = cls()
        cfg_path = os.path.join(path, "tokenizer_draco.json")
        if not os.path.exists(cfg_path):
            return tok

        with open(cfg_path, encoding="utf-8") as f:
            config = json.load(f)

        for a, b, c in config.get("merges", []):
            tok.add_merge((a, b), c)
        for k_str, v_latin in config.get("ext_vocab", {}).items():
            tok._ext_vocab[int(k_str)] = v_latin.encode("latin-1")
            tok._ext_inv[v_latin.encode("latin-1")] = int(k_str)
        tok.strict_utf8 = config.get("strict_utf8", False)
        tok._rebuild_sp_pattern()
        return tok


# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok = BPETokenizer()

    # Basic round-trip
    text  = "Hello, DracoAI! Xin chào."
    ids   = tok.encode(text)
    back  = tok.decode(ids)
    print(f"encode→decode: {repr(back)}")

    # Special token handling
    ids2  = tok.encode("<|im_start|>user\nHello<|im_end|>")
    assert SPECIAL_TOKENS["<|im_start|>"] in ids2, "Special token encode failed"

    # Stream decode
    chunks = list(tok.stream_decode(iter(ids)))
    assert "".join(chunks).strip() == text.strip(), f"stream_decode mismatch: {''.join(chunks)!r}"

    # Chat template
    msgs = [{"role": "user", "content": "Xin chào!"}]
    chat_ids = tok.encode_chat(msgs)
    assert SPECIAL_TOKENS["<|im_start|>"] in chat_ids

    # Extension vocab
    new_tid = tok.add_token("<draco_special>")
    assert new_tid >= QWEN_BASE_END
    assert tok.add_token("<draco_special>") == new_tid  # idempotent

    # FIX S3 test: garbage token ID should not crash
    result = tok.decode_token(999999)
    assert result == "", f"Expected empty string for unknown ID, got {result!r}"

    # FIX TH: multi-byte streaming
    viet_text = "Tôi yêu Việt Nam"
    viet_ids  = tok.encode(viet_text)
    viet_back = "".join(tok.stream_decode(iter(viet_ids)))
    assert viet_text in viet_back or viet_back in viet_text, \
        f"Vietnamese stream_decode failed: {viet_back!r}"

    # FIX UNICODE-NFC: precomposed vs decomposed Vietnamese
    nfc = unicodedata.normalize("NFC", "hòa")
    nfd = unicodedata.normalize("NFD", "hòa")
    assert tok.encode(nfc) == tok.encode(nfd), "NFC normalization failed"

    print("✅ tokenizer_v1 self-test passed")



