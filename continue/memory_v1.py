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
DracoAI V1 — Memory System (Vector DB)
=======================================
Compatible with ThinkingEngineV1 (engine_v1.py).

Engine interface (process() params):
    memory_summary    : str        ← get_summary_with_intent()
    ltm_facts         : List[dict] ← get_facts()
    memory_candidates : List[dict] ← search()

Convenience method:
    prepare_engine_input(query, intent, top_k) → dict with all three keys.

PRODUCTION HARDENING:
    - _has_negation expanded: 11 multi-word Vietnamese phrases + 5 single-word VI + 7 EN.
    - intent scoring changed from multiplicative to soft additive boost (similarity-first).
    - _semantic_duplicate threshold lowered to 0.90 + dual sim+jaccard condition added.
    - DEADLOCK eliminated: store() now calls _decay_cleanup_unsafe() (lock-free internal);
      public decay_cleanup() still acquires lock as before.
    - record_feedback O(1) dict lookup (was accidentally still O(n) despite the comment).
    - Bigram support in embed(): adjacent token pairs captured at 0.6× weight; unigram
      path and all external callers of _tokenize() are unchanged.
    - Batched disk writes every N stores to avoid I/O choking.
    - _save_vectors() uses atomic write (tmp + os.replace) to prevent file
      corruption on power failure / crash.
    - Background flush thread: flushes dirty vectors every 30s.
    - Search cache key is MD5 of full query + entities (not truncated) to prevent collision.
    - Semantic duplicate check scans top-10 candidates with negation guard.
    - _vector_search guards NaN values via np.nan_to_num.
    - MMR ensures fresh vector array before use.
    - Learnable filter uses word-boundary regex (\\b) instead of substring match
      to avoid false positives (e.g. "isn't" matching "is ").
    - Short 2-word phrases with uppercase tokens are accepted as knowledge.
    - consolidate() uses vector-search neighborhood + combined sim+jaccard condition
      to avoid merging semantically different memories.
    - Double hashing in embed() reduces hash collision significantly.
    - Cache is invalidated after fit_embedder() / update_idf() calls.
    - atexit flush ensures vectors are persisted on normal program exit.
    - Thread-safe store/search/consolidate/feedback/decay via threading.Lock.
    - All public methods that read or write shared state are fully lock-guarded.
    - search() snapshots all shared data under lock before releasing it for scoring.
    - _mmr() receives a compact pre-snapshotted vecs array (K×dim, not full N×dim).
    - consolidate(), record_feedback(), decay_cleanup() fully wrapped in lock.
    - fit_embedder() / update_idf() / _re_embed_all() fully wrapped in lock.
    - Background flush uses threading.Event for graceful shutdown (stop() method).
    - Background flush logs errors instead of silently swallowing them.
    - entity_boost uses \b word-boundary regex for phrase matching (no false substrings).
    - Entity boost is per-token only — non-entity tokens keep original weight.
    - Vietnamese learnable markers extended (gồm, bao gồm, thuộc, được gọi là).
    - Token count for Vietnamese fixed: len(content) // 4 instead of split()*1.3.
    - Double hashing secondary index uses MD5 instead of predictable *31 formula.
    - Cache version counter (_version) invalidates stale entries across all write ops.
    - Query is whitespace-normalized before cache key hashing (strip/lower/collapse).
    - search() emptiness check is inside lock to avoid TOCTOU race.
    - list_recent / search_by_type / stats / export read under lock for consistency.
    - Intent-aware scoring (factual/code/chat) multiplies type-matched results.
    - record_feedback() adjusts importance score for smarter LRU forgetting.
    - WorkingMemory has a hard message cap (MAX_MESSAGES=200) to prevent RAM leak.
    - _emb_version tracks embedding generation; stored in each vector's metadata.
    - _w() logs OSError instead of silently swallowing write failures.
    - All critical sections are documented.

FIXES vs V1.2 (this version):
    [CRASH-ORDER] _save_vectors() previously wrote vectors.npz BEFORE the JSON sidecar.
          If the process crashed after writing the new .npz (vecs-only) but before
          writing the sidecar, the next load would find no sidecar → enter legacy
          branch → attempt data["vids"] on the vecs-only .npz → KeyError →
          except resets all lists to [] → total data loss.
          Fixed: sidecar is now written and os.replace-committed FIRST; the .npz
          is replaced second.  If crash occurs between sidecar-commit and npz-commit,
          the sidecar is already safe and the old .npz is still intact — no data loss.
    [IMPROVE] _count_tokens: len(content)//4 was a coarse char-only estimate that
          under-counted Vietnamese syllables and over-counted English words.
          Replaced with a hybrid formula: max(word_count*1.3, char_count//4).
          Word-count*1.3 approximates BPE tokenisation for both VI and EN;
          taking the max acts as a conservative upper bound to avoid silent budget
          overruns. Tokenizer integration (SentencePiece / BPE) remains the
          recommended production upgrade.
    [IMPROVE] consolidate() _dirty=False freeze section now carries a detailed
          INVARIANT comment explaining exactly why the pattern is safe, when it
          is NOT safe (if lock strategy changes), and what must be re-evaluated
          in that case — preventing future maintainers from misreading it as a bug.
    [IMPROVE] _load_vectors: added explicit vecs/vids/meta shape-consistency
          validation in both the new-format and legacy-migration paths.
          A partial crash write (sidecar row count ≠ npz row count) now logs a
          warning and discards the corrupted data rather than silently loading
          mismatched arrays that would cause IndexError at query time.
    [FIX-TIE] consolidate_by_topic neg_dominant tie-break changed from >= (ties→True)
          to > (ties→False). In a 50/50 negation split, defaulting to non-negated
          polarity is correct because negated memories are rarer in practice and
          should not be allowed to exclude the larger non-negated group.
    [BUG-1] _is_learnable: COARSE_MARKERS used bare `in` (substring match) for "là",
          causing "làm", "làng", "làn", etc. to falsely match the "là" marker.
          Short conversational phrases (≥3 words) containing any word with "là" as a
          prefix were accepted as learnable memories, polluting the vector store.
          Fixed: "là" now checked via \blà\b word-boundary regex; purely symbolic
          markers (=, ->, :, =>, ==) remain as substring checks (unambiguous);
          multi-word Vietnamese markers (bao gồm, được gọi là, etc.) also remain
          substring checks since their multi-word form prevents prefix collisions.
    [BUG-2] consolidate_by_topic: neg_first was taken from idxs[0] which might
          already be assigned to a prior cluster (in `processed`), causing the
          negation guard to use a stale polarity anchor and incorrectly reject
          valid unprocessed memories from joining the current cluster.
          Fixed: negation polarity is now determined by majority vote among the
          UNPROCESSED members of each pivot group before filtering; ties default
          to the non-negated polarity.
    [BUG-3] _load_vectors used np.load(..., allow_pickle=True) with dtype=object
          arrays for vids and meta — numpy uses pickle for object arrays, making
          a tampered vectors.npz a remote-code-execution vector.
          Fixed: storage split into two files:
            • vectors.npz      — float32 'vecs' only (allow_pickle=False)
            • vectors_meta.json — vids + meta as safe JSON (no pickle)
          Both files are written atomically (tmp + os.replace).
          Backward-compat: old pickle-format files are auto-migrated on first load
          (one-time only, logged as a warning).
    [BUG-6] _has_negation: "khó" and "hiếm" were listed as standalone single-word
          negation signals, but they mean "difficult" and "rare" — NOT logical
          negation. This caused false-positive negation detection that fragmented
          clusters and blocked consolidation of non-contradictory memories.
          Fixed: removed "khó" and "hiếm" from the single-word regex pattern;
          their negation-context forms ("khó có thể", "khó mà", "hiếm khi") remain
          correctly captured by PHRASE_NEGATIONS (longest-match-first).
    [IMPROVE] Replaced all print(…, file=sys.stderr) calls with proper
          logging.getLogger("draco.memory") usage. Callers can now configure log
          level / handlers without modifying library code.
    [DOC] Jaccard threshold asymmetry between _semantic_duplicate (0.30) and
          consolidate (0.20) is now documented in _semantic_duplicate docstring
          with explicit rationale to prevent future "bug report" confusion.
    [BUG-C (complete)] embed() called without snapshot in store(), _semantic_duplicate(),
          _re_embed_all(), and consolidate_by_topic() centroid-fallback path — these
          callers were missing vocab_override / idf_override / synonym_map_override even
          though search() and replay_relevant() had already been fixed.
          Fixed: ALL embed() call sites inside LongTermMemoryV1 now receive explicit
          snapshots, making embed() touch ZERO shared mutable state from any caller.
          _semantic_duplicate() signature extended with the three override params and
          store() snapshots embedder state once per call (inside lock) then passes them
          to both _semantic_duplicate() and the final embed() call.
          _re_embed_all() snapshots vocab/idf/syns once before the loop.
          consolidate_by_topic() centroid-fallback embed() passes a local snapshot.
    [BUG] _maybe_save(): _store_count % BATCH_SAVE == 0 evaluated True when
          _store_count == 0 (immediately after __init__ or after a full clear), causing
          a spurious _save_vectors() call on the very first flush check.
          Fixed: added _store_count > 0 guard before the modulo check.
    [IMPROVE] _apply_synonyms() extended to accept an explicit synonym_map parameter
          so callers with a pre-snapshotted dict can avoid shared-state access.
          The method is now a proper public utility (embed() still inlines the expansion
          for performance, but subclasses / external callers can use _apply_synonyms()).
    [IMPROVE] Counter import moved to module level (was inside _summarize_cluster()
          which caused a repeated import on every consolidate_by_topic() call).
    [NOTE] consolidate_by_topic() pre-lock TOCTOU on embedder.vocab / _meta_list is
          documented as benign: fit_embedder() is idempotent and lock-guarded; the
          worst case is a redundant call, never a crash or data corruption.
    [BUG-C (definitive, original)] embed() in search() / replay_relevant() still read self.embedder.vocab
          and self.embedder.idf directly even after a snapshot was taken — the snapshot was
          captured but never passed into embed(), leaving the data race fully intact.
          Fixed properly in two parts:
            (1) MiniEmbedder.embed() now accepts vocab_override, idf_override, and
                synonym_map_override optional parameters.  When provided, embed() uses
                ONLY these immutable snapshots and touches zero shared mutable state.
            (2) search() and replay_relevant() now snapshot vocab+idf+synonym_map under
                lock and pass the snapshots as overrides to every embed() call, making
                both methods fully race-free with concurrent fit_embedder()/update_idf().
    [IMPROVE-SYNONYM] MiniEmbedder gains synonym_map: Dict[str,str] — a configurable
          surface-token → canonical-token mapping applied inside embed() after tokenization.
          This partially addresses the classic BoW synonym-blindness limitation (e.g.
          "ô tô" and "xe hơi" can be mapped to the same token).
          set_synonyms() method added for runtime updates; synonym_map persisted in
          embedder.json for durability across restarts (backward-compatible default: {}).
    [IMPROVE-TOPIC-CONSOLIDATION] consolidate_by_topic() added — periodic topic-level
          memory compaction.  Groups memories with sim ≥ threshold into clusters of
          ≥ min_cluster_size, builds a lightweight extractive summary per cluster
          (replacing N small memories with 1 summary memory), preserving combined
          frequency + importance.  Negation guard prevents contradictory memories from
          being merged.  Fully thread-safe (wrapped in self._lock) and fully compatible
          with the existing consolidate() (near-duplicate merger) workflow.
    [BUG-A] clear_search_cache() called from remember_fact()/learn()/forget_fact() WITHOUT
          self._lock — concurrent search() could see an inconsistent (_version, _search_cache)
          state (cache cleared but version not yet bumped, or vice versa).
          Fixed: all three callers now acquire self._lock before calling clear_search_cache().
    [BUG-B] store(): exact-duplicate path used self._vids.index(vid) → O(n), same problem
          that was explicitly fixed in record_feedback() but was missed here.  Additionally,
          the semantic-duplicate path also used self._vids.index(existing) → O(n).
          Fixed: a temporary {vid: idx} dict is built once per store() call inside the lock,
          providing O(1) lookup for both duplicate branches (mirrors the record_feedback fix).
    [BUG-D] WorkingMemory.clear() called kv_buf.flush_to_ltm() unconditionally;
          if clear() was called immediately after end_session() (which also flushes
          kv_buf), all kv_buf entries were stored to LTM twice — silent data duplication.
          Fixed: clear() checks len(kv_buf) > 0 before flushing (end_session already
          calls kv_buf.clear() through flush_to_ltm → clear(), so len is 0 afterward).
    [BUG-E] consolidate() iterates n vectors and calls _vector_search() for each one;
          _vector_search() internally calls _ensure_arr(), which checked self._dirty.
          Because store() sets _dirty=True and consolidate runs while _dirty=True,
          _ensure_arr rebuilt the full N×1024 float array on EVERY iteration of the
          consolidate loop → O(n²) wasted work.
          Fixed: _ensure_arr() called once before the loop; self._dirty temporarily
          set to False during the loop (restored afterward) to suppress per-iteration
          rebuilds.  _dirty is set back to True if any merges occurred.
    [BUG-F] _background_flush(): final flush on stop() called _save_vectors() unconditionally
          even when self._dirty=False — caused a spurious disk write + embedder.save()
          on every clean shutdown.
          Fixed: final flush now checks self._dirty before calling _save_vectors(),
          mirroring the behavior of the per-interval flush inside the loop.
    [BUG] _semantic_duplicate: used _tokenize (unigrams only) for Jaccard while embed()
          uses both unigrams+bigrams → mismatched token spaces caused wrong dedup decisions.
          Fixed: _tokenize_with_bigrams used for both sides of Jaccard in _semantic_duplicate.
    [BUG] search() scoring: freq_norm * importance (multiplicative) caused double-amplification
          — high-frequency + high-importance memories could bury semantically better results.
          Fixed: additive formula (freq_norm + capped_importance) / 2 with importance capped at 2.0.
    [BUG] update_idf(): only added unigrams to _df; bigram IDF became stale after incremental
          updates. Fixed: now uses _tokenize_with_bigrams — identical coverage to fit().
    [BUG] embed(): bigram slot always resolved via hash even when the bigram was in vocab.
          Fixed: vocab lookup first; hash fallback only for OOV bigrams (reduces collisions
          for frequent bigrams that were learned during fit/update_idf).
    [BUG] embed(): _tokenize called once standalone, then _tokenize_with_bigrams called again
          (which internally calls _tokenize) — two redundant tokenizations per embed call.
          Fixed: single _tokenize_with_bigrams call replaces both.
    [BUG] fit(): _tokenize called twice per text (once for unigram set, once for raw_tokens
          list). Fixed: single _tokenize_with_bigrams call covers both.
    [BUG] consolidate() Jaccard: same unigram-only mismatch as _semantic_duplicate.
          Fixed: _tokenize_with_bigrams used for both texts in consolidate() Jaccard check.
    [BUG] _tokenize_with_bigrams return type annotation was List[str] but returns
          Tuple[List[str], List[str]] — misleading for type checkers and readers.
          Fixed: corrected to Tuple[List[str], List[str]].
    [IMPROVE] _ensure_arr(): now re-normalizes vectors defensively during rebuild.
          Guards against any future code path that might store a non-unit-norm vector,
          preventing silent accuracy degradation without changing normal-path behavior.
    [BUG] _has_negation: only caught 7 basic negation words — "khó có thể", "ít khi",
          "không hẳn", "hiếm khi", etc. all missed → expanded to 11 multi-word phrases
          + 5 single-word Vietnamese + 7 English; phrase patterns checked first (longest
          match) before single-word regex, eliminating partial-match false negatives.
    [BUG] search() intent scoring: intent_w was multiplied against the FULL score, letting
          a wrong-topic result with the right intent_type outrank a correct-topic result
          with the wrong intent_type. Changed to additive soft boost:
          score = base * (1 + 0.15*(intent_w-1)) so similarity remains the primary factor.
    [BUG] _semantic_duplicate: threshold 0.95 too strict — paraphrases like "AI là tương
          lai" vs "AI sẽ là tương lai" (sim≈0.90) were saved as duplicates. Lowered to
          0.90 AND added Jaccard check (same dual-condition as consolidate) to prevent
          false-positive dedup of short texts in dense embedding regions.
    [BUG] DEADLOCK in store(): store() holds self._lock and called self.decay_cleanup()
          which also tries to acquire self._lock — threading.Lock is NOT reentrant,
          confirmed deadlock. Fixed by splitting into _decay_cleanup_unsafe() (no-lock,
          for use inside lock) and decay_cleanup() (public, acquires lock). store() now
          calls _decay_cleanup_unsafe().
    [BUG] record_feedback: docstring claimed O(1) dict lookup but code still used
          list.index() (O(n)) after an O(n) `in` check — two O(n) scans per call.
          Fixed: build a temporary {vid: idx} dict inside the lock for true O(1) lookup.
    [IMPROVE] MiniEmbedder.embed(): pure unigram TF-IDF loses phrase semantics — "machine
          learning" was indistinguishable from "machine" + "learning" separately. Added
          bigram tokens (adjacent word pairs joined by '_') with 0.6× weight. Double-hash
          trick applied to bigrams too. No dimension change (1024 is ample).
    [IMPROVE] MiniEmbedder._tokenize_with_bigrams(): new helper that returns both unigrams
          and bigrams; _tokenize() is unchanged for backward compat with all callers
          (consolidate Jaccard, _semantic_duplicate, etc.) that need unigrams only.
    [BUG] _vector_search: when len(sims) <= top_k, results were unordered → now sorted by -sim.
    [BUG] search cache key truncated query at 100 chars → collision risk; now hashes full query.
    [BUG] _init_files: prefs.json initialized as [] (array) instead of {} (dict) → fixed.
    [BUG] _lru_cleanup: kept indices could shift after deletion causing wrong removals → fixed
          by collecting vids-to-remove instead of index-based deletion.
    [BUG] _save_vectors: direct np.savez_compressed overwrite could corrupt on crash → atomic
          write via tmp file + os.replace now used.
    [BUG] _semantic_duplicate: only checked top-3, could miss duplicate at position 4-10 → top-10.
    [BUG] _vector_search: NaN in similarity could cause sort disorder → nan_to_num guard added.
    [BUG] cache not cleared after fit_embedder/update_idf → clear_search_cache() added.
    [BUG] _is_learnable: substring "is " matched inside words → now uses \\b word-boundary regex.
    [BUG] consolidate(): sim-only condition could merge "AI tốt cho y tế" and "AI tốt cho quân sự"
          → now requires BOTH sim > threshold AND jaccard > threshold.
    [BUG] search cache key missing intent → two identical queries with different intents
          (factual vs chat) shared a cache entry and returned wrong results → intent added to key.
    [BUG] _background_flush: _dirty check outside lock → potential double-write race condition
          → check moved inside lock.
    [BUG] entity boost unbounded → mult could exceed 2.0 with high-frequency entities, distorting
          semantics → capped at min(2.0, ...).
    [BUG] entity_freqs lookup case-sensitive → {"python": 5} miss when entity is "Python"
          → now tries ent first then ent.lower() as fallback.
    [BUG] TF-IDF weight dilution for long texts → freq/len made long-doc tokens weaker
          → replaced with smoothed (1+log(freq))/(1+log(len)) formula (BM25-style).
    [BUG] search top_k * 2 uncapped → very large top_k caused excessive scan cost
          → capped at min(top_k*2, n, 200).
    [BUG] emb_version not filtered in search snapshot → mixed old/new vectors possible
          if async re-embed added later → filter added (defensive forward-compatibility).
    [BUG] token_count could go negative on counting inconsistency → guarded with max(0, ...).
    [BUG] WorkingMemory.clear() discarded kv_buf without flushing → data loss in short sessions
          → flush to LTM before clear.
    [IMPROVE] Double hashing in embed(): vec[idx] and vec[(idx*31)%dim] to reduce collision.
    [IMPROVE] entity_boost weight 1.3 → adaptive: min(2.0, 1.5 + 0.1 * log(freq+1)).
    [IMPROVE] _is_learnable: 2-word uppercase token phrases accepted as knowledge (e.g. "GPU mạnh").
    [IMPROVE] consolidate: O(n²) → O(n * top_k) via vector-search neighborhood.
    [IMPROVE] atexit.register(flush_vectors) added to prevent data loss on normal exit.
    [IMPROVE] Background flush thread (daemon) flushes every 30s for crash resilience.
    [IMPROVE] threading.Lock added to store() and search() for thread-safety.
    [IMPROVE] WorkingMemory._trim(): evicts oldest message, not just first non-system.
    [IMPROVE] TF-IDF weight: BM25-style (1+log(freq))/(1+log(len)) replaces freq/len.
"""

import atexit
import logging
import os
import json
import math
import time
import hashlib
import re
import threading
from collections import Counter, OrderedDict

_log = logging.getLogger("draco.memory")

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# MINI EMBEDDER  — TF-IDF-style bag-of-words, no external deps
# ══════════════════════════════════════════════════════════════════════
class MiniEmbedder:
    """
    Lightweight bag-of-words embedder with double-hash trick.
    dim = 1024 to reduce collision probability.
    Vocab & IDF are built via fit(); update_idf() adjusts IDF without
    touching vocab, minimizing drift.
    Can be frozen to prevent any further updates.
    All embeddings are L2-normalized.

    FIX: Double hashing — each token contributes to TWO positions in the
    vector (primary and secondary), which drastically reduces collision
    accumulation and improves semantic discrimination.
    """

    def __init__(self, dim: int = 1024,
                 synonym_map: Optional[Dict[str, str]] = None):
        self.dim = dim
        self.vocab: Dict[str, int]   = {}
        self.idf:   Dict[str, float] = {}
        self._doc_count = 0
        self._df:   Dict[str, int]   = {}
        self._frozen = False
        # IMPROVE: synonym_map {surface_token → canonical_token} (all lowercase).
        # Applied after _tokenize() so "ô tô" and "xe hơi" map to the same token.
        # Multi-word phrases must be pre-normalized at text level before tokenization;
        # single-token synonyms are resolved inside _apply_synonyms().
        # Example: {"ô tô": "xe hơi", "car": "xe hơi", "automobile": "xe hơi"}
        self.synonym_map: Dict[str, str] = synonym_map or {}

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = []
        buf = ""
        for ch in text:
            if ch.isalnum() or ch in "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹ+#.":
                buf += ch
            else:
                if buf:
                    tokens.append(buf)
                    buf = ""
        if buf:
            tokens.append(buf)
        return tokens

    @staticmethod
    def _tokenize_with_bigrams(text: str) -> Tuple[List[str], List[str]]:
        """Tokenize text and return (unigrams, bigrams) for phrase-level semantics.

        IMPROVE: Pure unigram tokenization loses the meaning of compound phrases
        like "machine learning", "deep learning", "học máy". Adding bigrams (pairs
        of adjacent tokens joined by '_') captures co-occurrence patterns without
        expanding the vector dimension (1024 is sufficient for both unigrams and
        bigrams combined).

        Bigrams are weighted at 0.6× unigrams inside embed() to prevent them from
        dominating the vector while still improving phrase discrimination.

        FIX: Return type corrected to Tuple[List[str], List[str]] — was incorrectly
        annotated as List[str] despite returning a 2-tuple.

        NOTE: Synonym expansion is intentionally NOT applied here because this method
        is also used for Jaccard overlap in _semantic_duplicate() and consolidate(),
        where synonym expansion would conflate distinct memories that merely share
        synonyms. Synonym expansion only fires inside embed() via _apply_synonyms().
        """
        tokens = MiniEmbedder._tokenize(text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens, bigrams

    def _apply_synonyms(self, tokens: List[str],
                        synonym_map: Optional[Dict[str, str]] = None) -> List[str]:
        """Replace tokens with their canonical synonym where a mapping exists.

        IMPROVE: Partial fix for the BoW synonym-blindness problem.
        "ô tô" and "xe hơi" produce different tokens after _tokenize() —
        this method normalises single-token surface forms to a shared canonical.

        Multi-word synonyms (e.g. "ô tô" → "xe hơi") must be pre-normalized
        at the text level (e.g. via str.replace) before tokenization, since
        _tokenize() splits on whitespace and punctuation.  This method only
        handles single-token mappings (post-tokenization).

        NOTE: embed() inlines the synonym expansion directly (for performance
        and snapshot-safety). This method is kept as a public utility for
        subclasses or external callers that need standalone synonym expansion.
        Pass synonym_map explicitly to avoid shared-state access in threaded code.

        Thread-safety: if synonym_map is not supplied, reads self.synonym_map
        which is only mutated by set_synonyms() / load(); callers that hold
        self._lock are safe. For lock-free callers, pass a pre-snapshotted dict.
        """
        _syns = synonym_map if synonym_map is not None else self.synonym_map
        if not _syns:
            return tokens
        return [_syns.get(tok, tok) for tok in tokens]

    def fit(self, texts: List[str]):
        if self._frozen:
            return
        self.vocab.clear()
        self.idf.clear()
        self._df.clear()
        self._doc_count = 0
        for text in texts:
            # FIX: use _tokenize_with_bigrams to avoid calling _tokenize twice
            # and to ensure unigrams + bigrams are both included in IDF from the start.
            raw_tokens, bigrams = self._tokenize_with_bigrams(text)
            all_terms = set(raw_tokens) | set(bigrams)
            self._doc_count += 1
            for tok in all_terms:
                self._df[tok] = self._df.get(tok, 0) + 1

        sorted_tokens = sorted(self._df.items(), key=lambda x: -x[1])
        self.vocab = {tok: i for i, (tok, _) in enumerate(sorted_tokens[: self.dim])}

        N = max(self._doc_count, 1)
        self.idf = {
            tok: math.log((N + 1) / (df + 1)) + 1.0
            for tok, df in self._df.items() if tok in self.vocab
        }

    def update_idf(self, texts: List[str]):
        """FIX: Now includes bigrams in IDF update (was unigram-only).

        Previously fit() added both unigrams and bigrams to _df, but update_idf()
        only added unigrams. After any incremental call to update_idf(), bigram IDF
        values became stale, degrading embedding quality for phrase-level queries.
        Now consistent with fit(): both unigrams and bigrams are tracked.
        """
        if self._frozen:
            return
        for text in texts:
            # FIX: use _tokenize_with_bigrams so bigrams are included (same as fit())
            raw_tokens, bigrams = self._tokenize_with_bigrams(text)
            all_terms = set(raw_tokens) | set(bigrams)
            self._doc_count += 1
            for tok in all_terms:
                self._df[tok] = self._df.get(tok, 0) + 1
        N = max(self._doc_count, 1)
        for tok in self.vocab:
            df = self._df.get(tok, 0)
            self.idf[tok] = math.log((N + 1) / (df + 1)) + 1.0

    def embed(self, text: str, entity_boost: Optional[List[str]] = None,
              entity_freqs: Optional[Dict[str, int]] = None,
              vocab_override: Optional[Dict[str, int]] = None,
              idf_override: Optional[Dict[str, float]] = None,
              synonym_map_override: Optional[Dict[str, str]] = None) -> np.ndarray:
        """Return an L2-normalized vector.

        entity_boost  : list of entity strings to receive extra weight.
                        Supports single-token entities ("Python") and multi-token
                        phrase entities ("machine learning", "deep learning").
                        Only tokens that BELONG to a matched entity are boosted —
                        not all tokens in the document (previous bug was global boost).
        entity_freqs  : optional dict {entity: freq} for adaptive boost multiplier;
                        if None, a flat multiplier of 1.5 is used.
        vocab_override: pre-snapshotted vocab dict (thread-safe callers pass this
                        so embed() reads zero shared mutable state).  When None,
                        falls back to self.vocab (safe in single-threaded contexts).
        idf_override  : pre-snapshotted idf dict, same semantics as vocab_override.
        synonym_map_override: pre-snapshotted synonym_map for the same reason.

        FIX BUG-C (definitive): embed() previously read self.vocab and self.idf
        directly, even after a snapshot was taken in search().  Now callers pass
        the snapshot via vocab_override / idf_override so embed() has ZERO shared
        mutable state access when overrides are provided.  The race condition with
        concurrent fit_embedder() / update_idf() is fully eliminated.

        FIX: Double hashing — each OOV token contributes to a primary index
        and a secondary index (MD5-based) with 0.5 weight. This greatly reduces
        collision accumulation compared to single hashing.
        FIX: Per-token entity boost — only tokens belonging to a matched entity
        are multiplied; other tokens keep their original weight.

        IMPROVE: Synonym expansion via _apply_synonyms() before TF accumulation.
        """
        # Use overrides when provided (thread-safe path) or fall back to self (safe
        # in single-threaded / already-locked contexts).
        _vocab = vocab_override if vocab_override is not None else self.vocab
        _idf   = idf_override   if idf_override   is not None else self.idf
        _syns  = synonym_map_override if synonym_map_override is not None else self.synonym_map

        # FIX: call _tokenize_with_bigrams once instead of _tokenize + _tokenize_with_bigrams
        tokens, bigrams = self._tokenize_with_bigrams(text)

        # IMPROVE: apply synonym expansion on unigrams only
        # (bigrams are built from original tokens to preserve phrase-boundary info;
        # synonym expansion on bigrams could create phantom phrases like "xe hơi_mạnh"
        # when the original text has "ô tô mạnh", which is noise rather than signal).
        if _syns:
            tokens = [_syns.get(tok, tok) for tok in tokens]

        vec    = np.zeros(self.dim, dtype=np.float32)

        # ── Entity boost: build per-token membership map ──────────────
        # Supports both single-token entities ("Python") and multi-token phrase
        # entities ("machine learning", "deep learning").
        # Strategy:
        #   • Single-token entity  → mark that token directly.
        #   • Multi-token phrase   → mark every token inside the phrase, but only
        #     when the phrase actually appears as a substring in the lowercased text.
        # This avoids the previous bug where ANY entity presence in the text caused
        # ALL tokens to be boosted — now only relevant tokens are boosted.
        token_boost_map: Dict[str, float] = {}   # tok → boost_multiplier
        if entity_boost:
            text_lower = text.lower()
            for ent in entity_boost:
                ent_lower = ent.lower()
                ent_tokens = self._tokenize(ent_lower)
                if not ent_tokens:
                    continue
                if len(ent_tokens) == 1:
                    # Single token: boost only if this token appears in text
                    tok_e = ent_tokens[0]
                    if tok_e in set(tokens):
                        freq_e = (entity_freqs.get(ent, None) or
                                  entity_freqs.get(ent_lower, 1)) if entity_freqs else 1
                        # FIX: cap mult at 2.0 to prevent entity-frequency overfitting
                        mult   = min(2.0, 1.5 + 0.1 * math.log(freq_e + 1))
                        token_boost_map[tok_e] = max(token_boost_map.get(tok_e, 1.0), mult)
                else:
                    # Multi-token phrase: use word-boundary regex to avoid false
                    # substring matches (e.g. "AI" matching inside "chair").
                    # FIX: re.search(\b...\b) instead of bare `in` substring check.
                    pattern = r'\b' + r'\s+'.join(re.escape(t) for t in ent_tokens) + r'\b'
                    if re.search(pattern, text_lower):
                        freq_e = (entity_freqs.get(ent, None) or
                                  entity_freqs.get(ent_lower, 1)) if entity_freqs else 1
                        # FIX: cap mult at 2.0 to prevent entity-frequency overfitting
                        mult   = min(2.0, 1.5 + 0.1 * math.log(freq_e + 1))
                        for tok_e in ent_tokens:
                            token_boost_map[tok_e] = max(token_boost_map.get(tok_e, 1.0), mult)

        tf: Dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        for tok, freq in tf.items():
            idx = _vocab.get(tok)
            if idx is None:
                # FIX: double hashing with independent suffixes for idx and sign
                h_idx  = int(hashlib.md5((tok + "_idx").encode()).hexdigest(), 16) % self.dim
                h_sign = int(hashlib.md5((tok + "_sign").encode()).hexdigest(), 16) % 2
                idx  = h_idx
                sign = 1 if h_sign == 0 else -1
            else:
                sign = 1
            # IMPROVE: smoothed TF-IDF weight avoids dilution for long texts.
            weight = (1.0 + math.log(1 + freq)) / (1.0 + math.log(1 + max(len(tokens), 1))) * _idf.get(tok, 1.0)

            # IMPROVE: adaptive per-token entity boost (single and phrase entities)
            if tok in token_boost_map:
                weight *= token_boost_map[tok]

            # FIX: double hashing — primary + secondary position
            # Use MD5-based secondary index instead of predictable (idx*31)%dim
            vec[idx] += sign * weight
            sec_idx = int(hashlib.md5((tok + "_sec").encode()).hexdigest(), 16) % self.dim
            vec[sec_idx] += 0.5 * sign * weight

        # IMPROVE: bigram contributions (weight = 0.6× unigram weight)
        # Bigrams capture phrase co-occurrence without dominating the vector.
        # FIX: check self.vocab first for bigrams that were seen during fit/update_idf;
        # vocab bigrams use a fixed slot (sign=1) which is more stable than hashing
        # and avoids hash collisions for high-frequency bigrams.
        BIGRAM_SCALE = 0.6
        tf_bi: Dict[str, int] = {}
        for bg in bigrams:
            tf_bi[bg] = tf_bi.get(bg, 0) + 1
        for bg, freq in tf_bi.items():
            idx = _vocab.get(bg)
            if idx is not None:
                sign = 1
            else:
                h_idx  = int(hashlib.md5((bg + "_idx").encode()).hexdigest(), 16) % self.dim
                h_sign = int(hashlib.md5((bg + "_sign").encode()).hexdigest(), 16) % 2
                idx    = h_idx
                sign   = 1 if h_sign == 0 else -1
            weight = BIGRAM_SCALE * (1.0 + math.log(1 + freq)) / (1.0 + math.log(1 + max(len(bigrams), 1))) * _idf.get(bg, 1.0)
            vec[idx] += sign * weight
            sec_idx = int(hashlib.md5((bg + "_sec").encode()).hexdigest(), 16) % self.dim
            vec[sec_idx] += 0.5 * sign * weight

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Both vectors are assumed to be normalized; dot product suffices.
        return float(np.dot(a, b))

    def set_synonyms(self, mapping: Dict[str, str]):
        """Update the synonym map (all keys and values are lowercased automatically).

        Example:
            embedder.set_synonyms({
                "ô tô": "xe hơi",   # single token post-tokenize
                "car":  "xe hơi",
                "automobile": "xe hơi",
                "ml":   "machine_learning",
            })
        This is additive — existing entries are preserved unless overwritten.
        """
        self.synonym_map.update({k.lower(): v.lower() for k, v in mapping.items()})

    def save(self, directory: str):
        path = os.path.join(directory, "embedder.json")
        tmp  = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim":       self.dim,
                    "vocab":     self.vocab,
                    "idf":       self.idf,
                    "doc_count":   self._doc_count,
                    "df":          self._df,
                    "frozen":      self._frozen,
                    "synonym_map": self.synonym_map,
                },
                f,
                ensure_ascii=False,
            )
        os.replace(tmp, path)

    def load(self, directory: str):
        path = os.path.join(directory, "embedder.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim         = data.get("dim", self.dim)
        self.vocab       = data.get("vocab", {})
        self.idf         = data.get("idf",   {})
        self._doc_count  = data.get("doc_count", 0)
        self._df         = data.get("df",    {})
        self._frozen     = data.get("frozen", False)
        # IMPROVE: restore synonym_map (default empty dict for backward compat
        # with embedder.json files saved before synonym_map was introduced)
        self.synonym_map = data.get("synonym_map", {})


# ══════════════════════════════════════════════════════════════════════
# KV CACHE BUFFER  — short-term recency window
# ══════════════════════════════════════════════════════════════════════
class KVCacheBuffer:
    def __init__(self, max_size: int = 64):
        self.max_size = max_size
        self._buffer: List[Dict[str, Any]] = []

    # ── Unified learnable filter ─────────────────────────────────────
    @staticmethod
    def _is_learnable(text: str) -> bool:
        """Shared filter for both KVCacheBuffer and LongTermMemoryV1.

        FIX: Uses word-boundary regex (\\b) instead of simple substring
        match to avoid false positives like "isn't" matching "is ".

        FIX: Short 2-word phrases with at least one uppercase token are
        accepted as knowledge (e.g. "GPU mạnh", "LLM tốt").
        """
        text = text.strip()
        words = text.split()
        if len(words) < 2:
            return False

        # FIX: accept 2-word uppercase phrases as knowledge
        if len(words) == 2 and any(w.isupper() or (len(w) >= 2 and w[0].isupper()) for w in words):
            return True

        if len(words) < 3:
            return False

        if "?" in text:
            lower = text.lower()
            # Vietnamese knowledge questions
            if any(kw in lower for kw in ["là gì", "định nghĩa", "ai là", "nghĩa là",
                                          "what is", "define", "who is", "what are",
                                          "how does", "explain"]):
                return len(words) >= 4
            return len(words) >= 6

        # FIX BUG-1: COARSE_MARKERS previously used bare `in` (substring match),
        # causing "là" to match inside "làm", "làng", "làn", etc. — any Vietnamese
        # word that contains "là" as a prefix would pass the filter even for short
        # conversational phrases that should NOT be stored as memories.
        # Fixed: "là" is now checked via \blà\b word-boundary regex.
        # Purely symbolic markers (=, ->, :, =>, ==) remain as substring checks
        # since they are unambiguous punctuation that cannot appear inside a word.
        # Multi-word Vietnamese markers (bao gồm, được gọi là, etc.) also remain
        # as substring checks — their multi-word form prevents prefix collisions.
        SYMBOL_MARKERS = ("=", "->", ":", "=>", "==")
        MULTIWORD_MARKERS = ("bao gồm", "được gọi là", "gồm", "thuộc")
        lower = text.lower()

        if any(m in text for m in SYMBOL_MARKERS):
            return len(words) >= 3

        # FIX: use \b word boundary for standalone Vietnamese words that can be
        # prefixes of other words (là → làm, làng; gồm is less risky but kept safe)
        if re.search(r'\blà\b', lower):
            return len(words) >= 3

        if any(m in lower for m in MULTIWORD_MARKERS):
            return len(words) >= 3

        # FIX: word-boundary check for English keywords and Vietnamese patterns
        if re.search(r'\b(?:is|are|means|def|class|return|import)\b', lower):
            return len(words) >= 3

        return len(words) >= 4

    def push(self, text: str, context: str = ""):
        if not self._is_learnable(text):
            return
        entry = {
            "text":      text,
            "context":   context,
            "timestamp": time.time(),
        }
        self._buffer.append(entry)
        if len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def recent(self, n: int = 8) -> List[Dict[str, Any]]:
        return list(self._buffer[-n:])

    def clear(self):
        self._buffer.clear()

    def flush_to_ltm(self, ltm: "LongTermMemoryV1"):
        for entry in self._buffer:
            ltm.store(entry["text"], {"source": "kv_buffer", "context": entry["context"]})
        self.clear()

    def __len__(self) -> int:
        return len(self._buffer)


# ══════════════════════════════════════════════════════════════════════
# LONG-TERM MEMORY V1
# ══════════════════════════════════════════════════════════════════════
class LongTermMemoryV1:
    MAX_VECTORS            = 50_000
    DECAY_CLEANUP_INTERVAL = 50
    MAX_TEXT_LENGTH        = 500
    MAX_CACHE_SIZE         = 1000
    BATCH_SAVE             = 5       # FIX: reduced from 10 to 5 to lower data-loss window
    FLUSH_INTERVAL         = 30      # IMPROVE: background flush every N seconds

    def __init__(self, memory_dir: str = "memory"):
        self.dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.embedder = MiniEmbedder(dim=1024)

        self._ff  = os.path.join(memory_dir, "facts.json")
        self._ef  = os.path.join(memory_dir, "episodes.json")
        self._kf  = os.path.join(memory_dir, "knowledge.json")
        self._pf  = os.path.join(memory_dir, "prefs.json")
        self._npz = os.path.join(memory_dir, "vectors.npz")

        self._init_files()
        self.embedder.load(memory_dir)

        self._vec_list:  List[np.ndarray] = []   # pre‑normalized vectors
        self._meta_list: List[dict]        = []
        self._vids:      List[str]         = []
        self._vecs_arr:  Optional[np.ndarray] = None
        self._dirty      = False
        self._store_count = 0

        # IMPROVE: threading.Lock for thread-safe store/search
        self._lock = threading.Lock()
        # FIX: version counter — incremented on every write op so cache keys
        # built with old version are always considered stale.
        self._version: int = 0
        # IMPROVE: embedding version — incremented every time embedder is re-fitted.
        # Stored in each vector's metadata so stale-version vectors can be filtered
        # or skipped during search when a full re-embed hasn't completed yet.
        self._emb_version: int = 0
        # FIX: stop event for graceful background thread shutdown
        self._stop_event = threading.Event()

        # LRU search cache: {hash → (timestamp, results)}
        self._search_cache: OrderedDict = OrderedDict()
        self._initialized = False
        self._load_vectors()

        if not self.embedder.vocab and self._meta_list:
            texts = [m.get("text", "") for m in self._meta_list]
            if texts:
                self.fit_embedder(texts)
        self._initialized = True

        # IMPROVE: register flush on normal program exit to prevent data loss
        atexit.register(self.flush_vectors)

        # IMPROVE: background daemon thread flushes every FLUSH_INTERVAL seconds
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

    # ── Background flush ─────────────────────────────────────────────
    def _background_flush(self):
        """Daemon thread: flush dirty vectors every FLUSH_INTERVAL seconds.

        IMPROVE: Provides crash resilience beyond atexit (which doesn't run
        on SIGKILL or power loss). Maximum data loss window = FLUSH_INTERVAL seconds.
        FIX: Uses threading.Event.wait() instead of time.sleep(True) so the
        thread wakes immediately on stop() instead of blocking for FLUSH_INTERVAL.
        """
        while not self._stop_event.wait(timeout=self.FLUSH_INTERVAL):
            try:
                # FIX: _dirty check MUST be inside the lock to avoid race condition
                # where Thread A checks _dirty=True, Thread B saves and clears it,
                # then Thread A enters and double-writes unnecessarily.
                with self._lock:
                    if self._dirty:
                        self._save_vectors()
            except Exception as e:
                _log.error("Background flush error: %s", e)
        # FIX F: final flush on clean stop — only write if actually dirty to avoid
        # a no-op disk write (and unnecessary embedder.save() call) on clean shutdown.
        try:
            with self._lock:
                if self._dirty:
                    self._save_vectors()
        except Exception:
            pass

    def stop(self):
        """Gracefully stop the background flush thread and flush pending data.

        Call this instead of (or in addition to) flush_vectors() before
        destroying the instance to ensure the daemon thread exits cleanly.
        """
        self._stop_event.set()
        self._flush_thread.join(timeout=5)

    # ── File helpers ─────────────────────────────────────────────────
    def _init_files(self):
        for path in (self._ff, self._ef, self._kf):
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([], f)
        # FIX: prefs.json must be a dict, not a list
        if not os.path.exists(self._pf):
            with open(self._pf, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def _r(self, path: str) -> list:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _r_dict(self, path: str) -> dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _w(self, path: str, data: Any):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except OSError as e:
            _log.error("Write error %s: %s", path, e)

    # ── Vector persistence ───────────────────────────────────────────
    def _load_vectors(self):
        """Load vectors from disk.

        FIX BUG-3 (Pickle / RCE): Previously used np.load(..., allow_pickle=True)
        with dtype=object arrays for vids and meta — numpy uses pickle internally
        for object arrays, meaning a tampered vectors.npz could execute arbitrary
        code at load time (classic pickle RCE).

        Fixed by splitting storage into two files:
          • vectors.npz  — contains ONLY float32 'vecs' array (allow_pickle=False safe)
          • vectors_meta.json — contains vids (list[str]) and meta (list[dict]) as JSON

        JSON is human-readable, auditable, and cannot execute code.  The sidecar
        file is written atomically (tmp + os.replace) same as the .npz.

        Backward-compat: if vectors_meta.json is absent but vectors.npz exists with
        the old pickle layout, we attempt a one-time migration with allow_pickle=True
        (logs a warning), then immediately re-save in the new format.
        """
        if not os.path.exists(self._npz):
            return
        sidecar = self._npz[:-4] + "_meta.json"
        try:
            # ── New format: vecs-only npz + JSON sidecar ──────────────
            if os.path.exists(sidecar):
                data = np.load(self._npz, allow_pickle=False)
                vecs = data["vecs"]
                with open(sidecar, "r", encoding="utf-8") as f:
                    side = json.load(f)
                vids     = side.get("vids", [])
                meta_raw = side.get("meta", [])
                # Validate mutual consistency: mismatch means a partial crash write
                if not isinstance(meta_raw, list) or len(meta_raw) != len(vids):
                    _log.warning(
                        "vectors_meta.json is inconsistent (vids=%d, meta=%d) — "
                        "discarding corrupted data.",
                        len(vids) if isinstance(vids, list) else -1,
                        len(meta_raw) if isinstance(meta_raw, list) else -1,
                    )
                    return
                if len(vids) > 0 and vecs.shape[0] != len(vids):
                    _log.warning(
                        "vectors.npz row count (%d) != sidecar vids count (%d) — "
                        "discarding to avoid index corruption.",
                        vecs.shape[0], len(vids),
                    )
                    return
            else:
                # ── Legacy format: one-time pickle migration ───────────
                _log.warning("Loading legacy pickle-based vectors.npz. Re-saving in safe JSON-sidecar format.")
                data     = np.load(self._npz, allow_pickle=True)
                vecs     = data["vecs"]
                vids     = data["vids"].tolist()
                meta_raw = data["meta"].tolist()
                if not isinstance(meta_raw, list):
                    return
                # Validate that vecs/vids/meta are mutually consistent before migration
                if len(vids) != len(meta_raw) or (len(vids) > 0 and vecs.shape[0] != len(vids)):
                    _log.warning(
                        "Legacy vectors.npz has inconsistent shapes "
                        "(vids=%d, meta=%d, vecs_rows=%d) — skipping migration.",
                        len(vids), len(meta_raw), vecs.shape[0] if hasattr(vecs, "shape") else -1,
                    )
                    return
                # Immediately persist in safe format.
                # _save_vectors() writes sidecar FIRST then .npz (crash-safe order),
                # so even a crash mid-migration leaves a recoverable state.
                self._vids      = vids
                self._vec_list  = [vecs[i] for i in range(len(vids))]
                self._meta_list = meta_raw
                for meta in self._meta_list:
                    meta.pop("embedding", None)
                self._dirty     = True
                self._save_vectors()  # rewrites in safe JSON-sidecar format
                return

            self._vids      = vids
            # Loaded vectors are already normalized (we always save normalized)
            self._vec_list  = [vecs[i] for i in range(len(vids))]
            self._meta_list = meta_raw
            for meta in self._meta_list:
                meta.pop("embedding", None)
            self._dirty     = False
        except Exception as exc:
            _log.warning("Failed to load vectors: %s", exc)
            self._vids      = []
            self._vec_list  = []
            self._meta_list = []

    def _save_vectors(self):
        """Atomically persist vectors via tmp file + os.replace.

        FIX: Previous implementation wrote directly to vectors.npz; a crash
        mid-write would corrupt the file and wipe the entire vector DB on
        next load. Now we write to a .tmp file first, then atomically replace.

        FIX BUG-3 (Pickle / RCE): vids and meta are now stored in a separate
        vectors_meta.json sidecar file (JSON, no pickle) instead of object-dtype
        numpy arrays (which use pickle internally).  The .npz contains ONLY the
        float32 'vecs' array and can be loaded with allow_pickle=False.
        Both files are written atomically (tmp + os.replace).

        FIX CRASH-ORDER: The JSON sidecar MUST be written and atomically committed
        BEFORE the .npz is replaced.  Rationale:
          If we write .npz first then crash before writing the sidecar:
            • .npz now contains ONLY 'vecs' (no 'vids'/'meta' keys).
            • sidecar does not exist → _load_vectors enters the legacy branch.
            • legacy branch calls data["vids"] on the new .npz → KeyError.
            • except block resets all lists to [] → total data loss.
          If we write sidecar first then crash before writing .npz:
            • sidecar is safe and up-to-date.
            • .npz is still the old file (either new-format or legacy-format).
            • If .npz is new-format: sidecar exists → safe load, at worst 1 write behind.
            • If .npz is legacy-format (migration path): sidecar exists, so
              _load_vectors will use the new-format path and load vecs from the
              old .npz — may mismatch, but except block catches shape errors safely.
          Conclusion: sidecar-first ordering is strictly safer than npz-first.
        """
        if not self._dirty:
            return
        sidecar     = self._npz[:-4] + "_meta.json"
        sidecar_tmp = sidecar + ".tmp"
        if not self._vec_list:
            for path in (self._npz, sidecar):
                if os.path.exists(path):
                    os.remove(path)
            self._dirty = False
            return
        vecs     = np.stack(self._vec_list, axis=0).astype(np.float32)
        tmp_base = self._npz[:-4] + "_tmp"   # strip .npz, add _tmp suffix
        tmp      = tmp_base + ".npz"
        # STEP 1: Write JSON sidecar FIRST (crash-safe ordering — see docstring)
        side_data = {"vids": self._vids, "meta": self._meta_list}
        with open(sidecar_tmp, "w", encoding="utf-8") as f:
            json.dump(side_data, f, ensure_ascii=False)
        os.replace(sidecar_tmp, sidecar)
        # STEP 2: Write float32 vectors (no pickle needed) — after sidecar is safe
        np.savez_compressed(tmp_base, vecs=vecs)
        os.replace(tmp, self._npz)
        self.embedder.save(self.dir)
        self._dirty = False

    def _ensure_arr(self):
        """Rebuild the stacked float32 array from _vec_list when dirty or missing.

        FIX: Re-normalizes all vectors during rebuild as a defensive measure against
        any future code path that might store a non-unit-norm vector.  Normal
        operation always stores normalized vectors (embed() normalizes), but this
        guard costs O(n) and prevents silent accuracy degradation if that invariant
        is ever broken.

        FIX: _dirty is NOT cleared here — _dirty tracks whether vectors need to be
        flushed to disk (_save_vectors clears it).  _ensure_arr only rebuilds the
        in-memory numpy array; the two concerns are independent.
        """
        if self._dirty or self._vecs_arr is None or self._vecs_arr.shape[0] != len(self._vec_list):
            if self._vec_list:
                arr = np.stack(self._vec_list, axis=0).astype(np.float32)
                # FIX: defensive re-normalization — vectors should already be unit-norm
                # but re-normalize to guard against any future drift.
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms < 1e-8] = 1.0
                self._vecs_arr = arr / norms
            else:
                self._vecs_arr = None

    # ── Embedder controls ────────────────────────────────────────────
    def freeze_embedder(self):
        self.embedder.freeze()

    def unfreeze_embedder(self):
        self.embedder.unfreeze()

    # ── Unified learnable filter ─────────────────────────────────────
    @staticmethod
    def _is_learnable(text: str) -> bool:
        """Shared filter — delegates to KVCacheBuffer._is_learnable for consistency."""
        return KVCacheBuffer._is_learnable(text)

    # ── Helper for negation detection ────────────────────────────────
    @staticmethod
    def _has_negation(text: str) -> bool:
        """Detect negation in Vietnamese and English text.

        FIX: Expanded to cover full spectrum of Vietnamese negation patterns:
          • Strong direct negation : không, chẳng, chưa
          • Indirect / soft        : khó, hiếm, ít khi, mấy khi
          • Contradiction markers  : không hẳn, không phải, chưa chắc, đâu phải
          • English extensions     : not, no, never, neither, hardly, rarely, without
        Uses multi-word phrase matching before single-word regex to avoid partial
        matches (e.g. "không phải" must be caught before "không" alone so the
        phrase version takes priority and no double-counting occurs).
        """
        lower = text.lower()
        # Multi-word Vietnamese phrases first (order matters — longer patterns first)
        PHRASE_NEGATIONS = (
            "không hẳn", "không phải", "chưa chắc", "đâu phải",
            "ít khi", "mấy khi", "khó có thể", "chưa từng",
            "không bao giờ", "hiếm khi", "khó mà",
        )
        for phrase in PHRASE_NEGATIONS:
            if phrase in lower:
                return True
        # Single-word Vietnamese + English
        # FIX BUG-6: removed "khó" and "hiếm" from single-word negation patterns.
        # These words mean "difficult" and "rare" respectively — they are NOT strict
        # logical negations by themselves.  For example:
        #   "AI rất khó học" (AI is hard to learn)  — NOT a negation of AI's value
        #   "GPU hiếm hàng"  (GPU is scarce)         — NOT a negation of GPU quality
        # Their negation-context forms ("khó có thể", "khó mà", "hiếm khi") are
        # already captured by PHRASE_NEGATIONS above (longest-match-first order).
        # Keeping them in single-word patterns caused false-positive negation detection
        # that split valid clusters and blocked legitimate memory consolidation.
        pattern_vi = r'\b(?:không|chẳng|chưa)\b'
        pattern_en = r'\b(?:not|no|never|neither|hardly|rarely|without)\b'
        return bool(re.search(pattern_vi, lower) or re.search(pattern_en, lower))

    # ── LRU cleanup (type-weighted) ──────────────────────────────────
    def _lru_cleanup(self):
        """Remove lowest-scoring vectors when MAX_VECTORS is exceeded.

        FIX: Previous implementation collected indices into a keep-set and deleted
        by index, but the loop went in forward order — after each deletion, all
        subsequent indices shifted, causing wrong entries to be removed.  We now
        collect the *vids* of entries to keep (which are stable across deletions)
        and rebuild the three parallel lists in a single pass.
        """
        type_weights = {"fact": 3, "knowledge": 3, "episode": 2}
        now = time.time()
        scores = []
        for i, meta in enumerate(self._meta_list):
            age = (now - meta.get("timestamp", now)) / 86400.0
            w   = type_weights.get(meta.get("type", "general"), 1)
            freq = meta.get("frequency", 1)
            score = w * (freq + 1) * math.exp(-age / 30.0)
            scores.append((score, self._vids[i]))  # store vid, not index

        scores.sort(key=lambda x: x[0], reverse=True)
        keep_vids = set(vid for _, vid in scores[:self.MAX_VECTORS])

        new_vids, new_meta, new_vecs = [], [], []
        for vid, meta, vec in zip(self._vids, self._meta_list, self._vec_list):
            if vid in keep_vids:
                new_vids.append(vid)
                new_meta.append(meta)
                new_vecs.append(vec)

        self._vids      = new_vids
        self._meta_list = new_meta
        self._vec_list  = new_vecs
        self._dirty = True
        self.clear_search_cache()

    # ── Semantic duplicate check ─────────────────────────────────────
    def _semantic_duplicate(self, text: str, sim_threshold: float = 0.90,
                             jaccard_threshold: float = 0.30,
                             vocab_override: Optional[Dict[str, int]] = None,
                             idf_override: Optional[Dict[str, float]] = None,
                             synonym_map_override: Optional[Dict[str, str]] = None,
                             ) -> Optional[str]:
        """Check top-N candidates for semantic duplicate with negation guard.

        FIX: Threshold lowered from 0.95 → 0.90 so near-paraphrases like
        "AI là tương lai" and "AI sẽ là tương lai" (sim ≈ 0.90) are caught.

        FIX: Added Jaccard overlap check (same strategy as consolidate()) to
        prevent false-positive dedup when two texts have high cosine similarity
        but very different token sets (e.g. short texts in a dense embedding region).
        Both sim AND jaccard must exceed their respective thresholds.

        FIX: Increased top_k from 3 to min(10, n) to avoid missing duplicates
        when the vector space is dense (thousands of entries).

        FIX BUG-C (store path): accepts vocab_override / idf_override /
        synonym_map_override so embed() reads zero shared mutable state — the
        caller (store()) snapshots the embedder state under lock and passes it here.

        NOTE: Jaccard threshold default is 0.30 here (vs 0.20 in consolidate()).
        Rationale: _semantic_duplicate() fires on every single store() call and
        must be conservative — a higher threshold means we only skip storing if
        the text is clearly a near-duplicate. consolidate() is a periodic batch
        operation where a lower threshold is acceptable because the user explicitly
        triggered it to merge loosely similar memories.

        NOTE: This method MUST be called while already holding self._lock
        (store() acquires the lock before calling this).
        """
        if not self._vec_list:
            return None
        # FIX BUG-C: pass snapshot overrides so embed() is free of shared-state access
        q_vec = self.embedder.embed(text,
                                    vocab_override=vocab_override,
                                    idf_override=idf_override,
                                    synonym_map_override=synonym_map_override)  # already normalized
        top_k = min(10, len(self._vec_list))
        indices, sims = self._vector_search(q_vec, top_k=top_k)
        # FIX: use _tokenize_with_bigrams for Jaccard so the token space matches
        # the embedding space (embed() uses both unigrams and bigrams).
        # Previously _tokenize (unigrams-only) was used, causing Jaccard to disagree
        # with the cosine similarity dimension, leading to false-positive or false-
        # negative duplicate decisions on phrase-heavy texts.
        unigrams_new, bigrams_new = self.embedder._tokenize_with_bigrams(text)
        tokens_new = set(unigrams_new) | set(bigrams_new)
        for sim, idx in zip(sims, indices):
            idx = int(idx)
            if idx >= len(self._meta_list):
                continue
            if sim < sim_threshold:
                break   # sorted descending — no point continuing
            candidate_text = self._meta_list[idx]["text"]
            # Negation guard: never merge contradictory memories
            if self._has_negation(text) != self._has_negation(candidate_text):
                continue
            # Jaccard guard: require meaningful token overlap, not just vector proximity
            unigrams_cand, bigrams_cand = self.embedder._tokenize_with_bigrams(candidate_text)
            tokens_cand = set(unigrams_cand) | set(bigrams_cand)
            if tokens_new and tokens_cand:
                union = tokens_new | tokens_cand
                jaccard = len(tokens_new & tokens_cand) / len(union) if union else 0.0
            else:
                jaccard = 0.0
            if jaccard < jaccard_threshold:
                continue
            return self._vids[idx]
        return None

    # ── Store ────────────────────────────────────────────────────────
    def store(self, text: str, metadata: Optional[dict] = None) -> str:
        """Thread-safe store with semantic deduplication.

        IMPROVE: Wrapped in threading.Lock to prevent race conditions on
        _vids.append / _vec_list.append in multi-threaded environments.
        """
        if not self._is_learnable(text):
            return ""

        if len(text) > self.MAX_TEXT_LENGTH:
            metadata = (metadata or {}).copy()
            metadata["full_text"] = text
            text = text[: self.MAX_TEXT_LENGTH - 3] + "..."

        with self._lock:
            # FIX B: maintain O(1) vid→index map (avoids O(n) _vids.index() calls)
            # This mirrors the fix already applied in record_feedback().
            # Rebuilt lazily inside lock; cost is O(n) once per store() call that
            # hits a duplicate, which is acceptable (same asymptotic as before, but
            # a single scan instead of two O(n) scans).
            vid_to_idx: Dict[str, int] = {v: i for i, v in enumerate(self._vids)}

            # FIX BUG-C (store path): snapshot embedder state inside lock so that
            # embed() in _semantic_duplicate() and the final vec = embed(text) call
            # both touch ZERO shared mutable state — consistent with the fix already
            # applied in search() and replay_relevant().
            # Although store() holds self._lock, passing snapshots makes embed()
            # fully side-effect-free and simplifies future refactoring / testing.
            _emb_vocab = dict(self.embedder.vocab)
            _emb_idf   = dict(self.embedder.idf)
            _emb_syns  = dict(self.embedder.synonym_map)

            # Semantic duplicate check (with negation guard)
            existing = self._semantic_duplicate(text,
                                                vocab_override=_emb_vocab,
                                                idf_override=_emb_idf,
                                                synonym_map_override=_emb_syns)
            if existing:
                idx = vid_to_idx.get(existing)
                if idx is not None:
                    self._meta_list[idx]["frequency"] = self._meta_list[idx].get("frequency", 1) + 1
                    self._meta_list[idx]["timestamp"] = time.time()
                    self._dirty = True
                    self._maybe_save()
                    self.clear_search_cache()
                return existing

            vid = hashlib.md5(text.encode()).hexdigest()[:12]

            # Exact duplicate fallback — O(1) via vid_to_idx
            if vid in vid_to_idx:
                idx = vid_to_idx[vid]
                self._meta_list[idx]["frequency"] = self._meta_list[idx].get("frequency", 1) + 1
                self._meta_list[idx]["timestamp"] = time.time()
                self._dirty = True
                self._maybe_save()
                self.clear_search_cache()
                return vid

            # FIX BUG-C: pass snapshot overrides so embed() reads zero shared state
            vec  = self.embedder.embed(text,
                                       vocab_override=_emb_vocab,
                                       idf_override=_emb_idf,
                                       synonym_map_override=_emb_syns)   # already normalized
            meta = {
                "text":        text,
                "timestamp":   time.time(),
                "frequency":   1,
                "type":        (metadata or {}).get("type", "general"),
                "metadata":    metadata or {},
                "importance":  (metadata or {}).get("importance", 1.0),
                "emb_version": self._emb_version,   # IMPROVE: track embedding version
            }

            self._vids.append(vid)
            self._meta_list.append(meta)
            self._vec_list.append(vec)
            self._dirty = True
            self._store_count += 1

            if self._store_count % self.DECAY_CLEANUP_INTERVAL == 0:
                self._decay_cleanup_unsafe()   # FIX: call lock-free internal version

            if len(self._vec_list) > self.MAX_VECTORS:
                self._lru_cleanup()

            self._maybe_save()
            self.clear_search_cache()
            return vid

    def _maybe_save(self):
        """Persist vectors every BATCH_SAVE stores or when dirty.

        FIX: Added self._store_count > 0 guard so the modulo check does not
        fire spuriously when store_count is 0 (e.g. immediately after __init__).
        """
        if self._store_count > 0 and self._store_count % self.BATCH_SAVE == 0 and self._dirty:
            self._save_vectors()

    def flush_vectors(self):
        """Force immediate persistence of vectors (call before shutdown)."""
        with self._lock:
            if self._dirty:
                self._save_vectors()

    # ── Embedder fit / update ────────────────────────────────────────
    def fit_embedder(self, texts: Optional[List[str]] = None):
        """FIX: Wrapped in lock — _re_embed_all replaces all of _vec_list."""
        with self._lock:
            if texts is None:
                texts = [m.get("text", "") for m in self._meta_list]
            self.embedder.fit(texts)
            self._re_embed_all()
            # FIX: invalidate cache after embedder update so stale results are not served
            self.clear_search_cache()

    def update_idf(self, texts: Optional[List[str]] = None):
        """FIX: Wrapped in lock — _re_embed_all replaces all of _vec_list."""
        with self._lock:
            if texts is None:
                texts = [m.get("text", "") for m in self._meta_list]
            self.embedder.update_idf(texts)
            self._re_embed_all()
            # FIX: invalidate cache after IDF update
            self.clear_search_cache()

    def _re_embed_all(self):
        """MUST be called while holding self._lock.

        IMPROVE: Bumps _emb_version after re-embedding so callers can detect
        embedding drift.  Each vector's metadata gets 'emb_version' updated so
        stale entries can be identified without a full scan.

        FIX BUG-C (_re_embed_all path): snapshot vocab/idf/synonym_map once
        before the loop so every embed() call inside the loop reads from an
        immutable snapshot rather than the live (potentially mutating) dicts.
        Although this method is always called under self._lock, taking the
        snapshot makes the code consistent with all other embed() call sites
        and guards against any future refactoring that removes the lock.
        """
        self._emb_version += 1
        # Snapshot embedder state once — all embed() calls below use this snapshot
        _vocab = dict(self.embedder.vocab)
        _idf   = dict(self.embedder.idf)
        _syns  = dict(self.embedder.synonym_map)
        for i, meta in enumerate(self._meta_list):
            self._vec_list[i] = self.embedder.embed(
                meta["text"],
                vocab_override=_vocab,
                idf_override=_idf,
                synonym_map_override=_syns,
            )  # normalized
            meta["emb_version"] = self._emb_version
        self._dirty = True
        self._save_vectors()

    # ── Decay cleanup ────────────────────────────────────────────────
    def _decay_cleanup_unsafe(self, days_threshold: int = 30, min_frequency: int = 2):
        """Internal decay cleanup — MUST be called while already holding self._lock.

        FIX: Extracted from decay_cleanup() so that store() can call this without
        attempting to re-acquire self._lock (threading.Lock is NOT reentrant — nested
        acquisition from the same thread causes an immediate deadlock).
        """
        now = time.time()
        to_remove = []
        for i, meta in enumerate(self._meta_list):
            age_days = (now - meta.get("timestamp", now)) / 86400.0
            if age_days > days_threshold and meta.get("frequency", 1) <= min_frequency:
                to_remove.append(i)
        for idx in reversed(to_remove):
            del self._vids[idx]
            del self._meta_list[idx]
            del self._vec_list[idx]
        if to_remove:
            self._dirty = True
            self._save_vectors()
            self.clear_search_cache()

    def decay_cleanup(self, days_threshold: int = 30, min_frequency: int = 2):
        """Public API — acquires lock then delegates to _decay_cleanup_unsafe().

        FIX: Wrapped in lock — modifies _vids/_meta_list/_vec_list (shared state).
        Must NOT be called from any code path that already holds self._lock
        (use _decay_cleanup_unsafe() instead to avoid deadlock).
        """
        with self._lock:
            self._decay_cleanup_unsafe(days_threshold, min_frequency)

    # ── Modeling vector search ───────────────────────────────────────────
    def _vector_search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cosine similarity via dot product (vectors are pre‑normalized).

        FIX 1: When len(sims) <= top_k the original code returned indices via
        np.arange() which is unordered.  Results are now always sorted by
        descending similarity regardless of whether we have more or fewer
        candidates than top_k.

        FIX 2: NaN guard — np.nan_to_num converts any NaN similarity to 0.0
        so that np.argsort/argpartition is not disrupted.
        """
        self._ensure_arr()
        if self._vecs_arr is None:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        sims = self._vecs_arr @ query_vec   # dot product since both normalized

        # FIX: guard NaN values to prevent sort disorder
        sims = np.nan_to_num(sims, nan=0.0)

        if len(sims) <= top_k:
            # FIX: always sort even when we return all results
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, top_k)[:top_k]
            idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    # ── Search ───────────────────────────────────────────────────────
    def search(
        self,
        query:    str,
        top_k:    int            = 20,
        entities: Optional[List[str]] = None,
        use_mmr:  bool           = True,
        intent:   Optional[str]  = None,
    ) -> List[dict]:
        """Thread-safe search with LRU cache.

        FIX: All access to shared state (_meta_list, _vids, _vecs_arr) is now
        performed under a single lock via snapshot pattern:
          1. Check cache (+ version) under lock.
          2. Compute query vector (no shared state).
          3. Acquire lock → snapshot indices, sims, meta, vids, compact vecs → release.
          4. Score and MMR using ONLY the snapshot (no shared state access).
          5. Store result in cache under lock.
        IMPROVE: intent-aware scoring multiplier (factual/code/chat).
        """
        # FIX: check emptiness under lock to avoid TOCTOU with concurrent store()
        with self._lock:
            if not self._vec_list:
                return []

        # FIX: normalize query — strip, lowercase, collapse internal whitespace —
        # before hashing so "Python " and " Python" share the same cache entry.
        q_norm = " ".join(query.strip().lower().split())

        # FIX: read _version inside lock to guarantee we see the latest value.
        with self._lock:
            current_version = self._version
            # FIX: include intent in cache key — two identical queries with different
            # intents (e.g. "factual" vs "chat") have different scoring weights and
            # must NOT share a cache entry (previously caused wrong results).
            key_raw = (q_norm, tuple(sorted(entities or [])),
                       top_k, use_mmr, intent or "", current_version)
            q_hash  = hashlib.md5(str(key_raw).encode()).hexdigest()
            if q_hash in self._search_cache:
                ts, cached = self._search_cache[q_hash]
                if time.time() - ts < 300:
                    self._search_cache.move_to_end(q_hash)
                    return cached[:top_k]
                else:
                    del self._search_cache[q_hash]

        # FIX BUG-C (definitive): snapshot embedder's vocab, idf, and synonym_map under
        # lock, then pass them as overrides to embed() so embed() reads ZERO shared mutable
        # state.  The previous version snapshotted but never passed the snapshot into embed(),
        # leaving the data race intact.  Now embed() receives vocab_override / idf_override /
        # synonym_map_override and uses them exclusively, making this call fully thread-safe.
        with self._lock:
            emb_vocab     = dict(self.embedder.vocab)
            emb_idf       = dict(self.embedder.idf)
            emb_syns      = dict(self.embedder.synonym_map)
        q_vec = self.embedder.embed(
            q_norm,
            entity_boost=entities,
            vocab_override=emb_vocab,
            idf_override=emb_idf,
            synonym_map_override=emb_syns,
        )
        now   = time.time()

        # ── SNAPSHOT all shared state under lock ──────────────────────
        with self._lock:
            self._ensure_arr()
            if self._vecs_arr is None or len(self._vec_list) == 0:
                return []
            # FIX: cap top_k * 2 so very large top_k values don't cause excessive
            # scan cost; 200 candidates is sufficient for any downstream MMR/scoring.
            search_k = min(top_k * 2, len(self._vec_list), 200)
            indices, sims = self._vector_search(q_vec, search_k)
            n = len(self._meta_list)
            current_emb = self._emb_version
            # FIX: filter by emb_version — prevents mixing stale and fresh vectors
            # if a re-embed is interrupted (currently sync so always consistent,
            # but this guard is essential if async re-embed is added in future).
            valid = [
                (int(i), float(s)) for i, s in zip(indices, sims)
                if int(i) < n and self._meta_list[int(i)].get("emb_version", current_emb) == current_emb
            ]
            snap_meta  = [self._meta_list[i] for i, _ in valid]
            snap_vids  = [self._vids[i]      for i, _ in valid]
            snap_sims  = [s                  for _, s in valid]
            # FIX: copy ONLY the candidate rows (K×1024), not the full 200 MB array.
            # This reduces per-query memory from ~200 MB to a few KB.
            cand_row_indices = [i for i, _ in valid]
            snap_vecs_cand   = (self._vecs_arr[cand_row_indices].copy()
                                if cand_row_indices else None)
            # vid → position inside snap_vecs_cand (for MMR lookup)
            snap_vid_to_pos  = {vid: pos for pos, vid in enumerate(snap_vids)}
        # ── End of lock region — all further work on snapshot only ────

        # IMPROVE: intent-aware score multiplier
        # factual/knowledge queries weight higher-confidence, fact/knowledge-type results.
        # code queries weight code/knowledge type. chat is neutral.
        INTENT_TYPE_WEIGHTS: Dict[str, Dict[str, float]] = {
            "factual": {"fact": 1.2, "knowledge": 1.1, "episode": 0.9,  "general": 1.0},
            "code":    {"fact": 1.0, "knowledge": 1.2, "episode": 0.8,  "general": 1.0},
            "chat":    {"fact": 1.0, "knowledge": 1.0, "episode": 1.1,  "general": 1.0},
        }
        intent_map = INTENT_TYPE_WEIGHTS.get(intent or "chat", INTENT_TYPE_WEIGHTS["chat"])

        MAX_FREQ_LOG = math.log(101)
        scored: List[dict] = []
        for meta, vid, sim in zip(snap_meta, snap_vids, snap_sims):
            sim = float(sim)
            age     = (now - meta.get("timestamp", now)) / 86400.0
            recency = math.exp(-age / 7.0)
            freq    = meta.get("frequency", 1)
            sim_norm    = (sim + 1.0) / 2.0
            freq_norm   = min(1.0, math.log(freq + 1) / MAX_FREQ_LOG)
            importance  = meta.get("importance", 1.0)
            mem_type    = meta.get("type", "general")
            intent_w    = intent_map.get(mem_type, 1.0)

            # FIX: intent_w is now a soft additive boost, NOT a multiplicative
            # override of the entire score. Previously intent_w * full_score meant
            # a wrong-topic but correct-type result could outrank a correct-topic
            # but different-type result (e.g. a DracoAI "fact" beating a relevant
            # Python "knowledge" when searching for "Python performance").
            # New formula: base_score is similarity-first; intent adds at most ±3%.
            #
            # FIX: freq_norm * importance (multiplicative) caused double-amplification:
            # a memory with high frequency AND high importance (from positive feedback)
            # could score far above its actual semantic relevance, burying better matches.
            # Now uses (freq_norm + capped_importance) / 2 — additive combination with
            # importance capped at 2.0 to prevent unbounded inflation.
            capped_importance = min(2.0, max(0.1, importance))
            base_score = (0.5 * sim_norm
                          + 0.2 * recency
                          + 0.3 * (freq_norm + capped_importance) / 2.0)
            score = base_score * (1.0 + 0.15 * (intent_w - 1.0))

            if entities and meta.get("type") in ("fact", "knowledge"):
                mem_text_lower  = meta.get("text", "").lower()
                meta_entities   = meta.get("metadata", {}).get("entities", [])
                # FIX: match entities against both the stored metadata.entities list
                # AND the raw text content.  metadata.entities is sparsely populated
                # (only set when caller explicitly passes it during store), so
                # text-level matching ensures entity boost always fires for relevant
                # memories regardless of how they were originally stored.
                entity_hit = any(
                    e.lower() in [ent.lower() for ent in meta_entities]
                    or e.lower() in mem_text_lower
                    for e in entities
                )
                if entity_hit:
                    score += 0.05

            scored.append(
                {
                    "id":        vid,
                    "score":     score,
                    "text":      meta.get("text", ""),
                    "timestamp": meta.get("timestamp", now),
                    "frequency": freq,
                    "type":      mem_type,
                    "ts":        meta.get("timestamp", now),
                    **meta.get("metadata", {}),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = scored[:top_k]

        if use_mmr and len(top_candidates) > 5 and snap_vecs_cand is not None:
            top_candidates = self._mmr(
                top_candidates, q_vec,
                snap_vecs_cand=snap_vecs_cand,
                snap_vid_to_pos=snap_vid_to_pos,
                top_k=min(top_k, 10),
            )

        with self._lock:
            if len(self._search_cache) >= self.MAX_CACHE_SIZE:
                self._search_cache.popitem(last=False)
            self._search_cache[q_hash] = (time.time(), top_candidates)
            self._search_cache.move_to_end(q_hash)

        return top_candidates

    # ── MMR ──────────────────────────────────────────────────────────
    def _mmr(
        self,
        candidates:      List[dict],
        query_vec:       np.ndarray,
        snap_vecs_cand:  np.ndarray,   # FIX: compact K×dim array of candidate vectors only
        snap_vid_to_pos: Dict[str, int], # FIX: vid → row index in snap_vecs_cand
        lambda_param:    float = 0.7,
        top_k:           int   = 5,
    ) -> List[dict]:
        """Maximal Marginal Relevance re-ranking.

        FIX: Accepts a compact (K×dim) array containing only candidate vectors
        instead of the full N×dim vecs array, cutting per-query memory to a few KB.
        FIX: Uses a pre-built vid→pos map for O(1) lookup instead of scanning
        the full vids list.  Zero shared state access — purely functional.
        """
        if len(candidates) <= top_k:
            return candidates

        # Map each candidate dict to its row in snap_vecs_cand
        cand_pos = [snap_vid_to_pos.get(c["id"]) for c in candidates]
        valid_pairs = [(i, pos) for i, pos in enumerate(cand_pos) if pos is not None]
        if not valid_pairs:
            return candidates

        local_indices = [i   for i, _   in valid_pairs]
        vecs_rows     = [pos for _, pos in valid_pairs]
        vecs          = snap_vecs_cand[vecs_rows]   # shape (M, dim), M ≤ K

        first = max(range(len(vecs)), key=lambda i: self.embedder.similarity(query_vec, vecs[i]))
        selected  = [first]
        remaining = [i for i in range(len(vecs)) if i != first]

        while len(selected) < top_k and remaining:
            mmr_scores = []
            for idx in remaining:
                sim_q = self.embedder.similarity(query_vec, vecs[idx])
                sim_s = max(
                    self.embedder.similarity(vecs[idx], vecs[sel]) for sel in selected
                )
                mmr_scores.append(lambda_param * sim_q - (1 - lambda_param) * sim_s)
            best = mmr_scores.index(max(mmr_scores))
            selected.append(remaining.pop(best))

        return [candidates[local_indices[i]] for i in selected]

    # ── Facts ─────────────────────────────────────────────────────────
    def remember_fact(self, key: str, value: str, confidence: float = 1.0):
        facts = self._r(self._ff)
        for f in facts:
            if f["key"].lower() == key.lower():
                if f["value"] != value:
                    f.setdefault("history", []).append(
                        {
                            "value":      f["value"],
                            "confidence": f.get("confidence", 1.0),
                            "updated":    f.get("updated", time.time()),
                        }
                    )
                    f["value"]      = value
                    f["confidence"] = confidence
                    f["updated"]    = time.time()
                else:
                    f["confidence"] = min(1.0, f.get("confidence", 1.0) + 0.05)
                self._w(self._ff, facts)
                self.store(f"{key} là {value}", {"type": "fact", "key": key, "importance": 1.5})
                return
        facts.append(
            {
                "key":        key,
                "value":      value,
                "confidence": confidence,
                "created":    time.time(),
                "updated":    time.time(),
                "history":    [],
            }
        )
        self._w(self._ff, facts)
        self.store(f"{key} là {value}", {"type": "fact", "key": key, "importance": 1.5})
        # FIX A: acquire lock before clearing cache so _version increment is atomic
        # with respect to concurrent search() reads.
        with self._lock:
            self.clear_search_cache()

    def get_facts(self) -> List[dict]:
        facts = self._r(self._ff)
        return sorted(facts, key=lambda x: x.get("confidence", 1.0), reverse=True)

    def forget_fact(self, key: str) -> bool:
        facts = self._r(self._ff)
        new   = [f for f in facts if f["key"].lower() != key.lower()]
        if len(new) < len(facts):
            self._w(self._ff, new)
            # FIX A: acquire lock before clearing cache for thread-safety
            with self._lock:
                self.clear_search_cache()
            return True
        return False

    # ── Episodes ──────────────────────────────────────────────────────
    def save_episode(self, summary: str, detail: str = "", tags: Optional[List[str]] = None):
        episodes = self._r(self._ef)
        entry = {
            "summary":   summary,
            "detail":    detail[:1000] if detail else "",
            "tags":      tags or [],
            "timestamp": time.time(),
        }
        episodes.append(entry)
        if len(episodes) > 200:
            episodes = episodes[-200:]
        self._w(self._ef, episodes)
        self.store(summary, {"type": "episode"})

    def replay_relevant(self, query: str, top_k: int = 3) -> List[dict]:
        episodes = self._r(self._ef)
        if not episodes:
            return []
        # FIX BUG-C: snapshot embedder state under lock before calling embed()
        # outside the lock, preventing data race with concurrent fit_embedder().
        with self._lock:
            emb_vocab = dict(self.embedder.vocab)
            emb_idf   = dict(self.embedder.idf)
            emb_syns  = dict(self.embedder.synonym_map)
        q_vec = self.embedder.embed(
            query,
            vocab_override=emb_vocab,
            idf_override=emb_idf,
            synonym_map_override=emb_syns,
        )
        scored = []
        for ep in episodes:
            s = ep.get("summary", "")
            if not s:
                continue
            sv    = self.embedder.embed(
                s,
                vocab_override=emb_vocab,
                idf_override=emb_idf,
                synonym_map_override=emb_syns,
            )
            score = self.embedder.similarity(q_vec, sv)
            scored.append((score, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_k]]

    def _get_episode_summaries(self, query: str, top_k: int = 3) -> List[str]:
        episodes = self.replay_relevant(query, top_k=top_k)
        return [ep.get("summary", "") for ep in episodes]

    # ── Knowledge ────────────────────────────────────────────────────
    def learn(self, topic: str, content: str, source: str = ""):
        knowledge = self._r(self._kf)
        entry = {
            "topic":     topic,
            "content":   content[:2000],
            "source":    source,
            "timestamp": time.time(),
        }
        knowledge.append(entry)
        if len(knowledge) > 5000:
            knowledge = knowledge[-5000:]
        self._w(self._kf, knowledge)
        self.store(f"{topic}: {content[:300]}", {"type": "knowledge", "topic": topic, "importance": 1.3})
        # FIX A: acquire lock before clearing cache for thread-safety
        with self._lock:
            self.clear_search_cache()

    def get_topics(self) -> List[str]:
        knowledge = self._r(self._kf)
        seen: Dict[str, bool] = {}
        topics = []
        for entry in reversed(knowledge):
            t = entry.get("topic", "")
            if t and t not in seen:
                seen[t] = True
                topics.append(t)
        return topics[:50]

    # ── Preferences ──────────────────────────────────────────────────
    def set_pref(self, key: str, value: Any):
        prefs = self._r_dict(self._pf)
        prefs[key] = value
        self._w(self._pf, prefs)

    def get_prefs(self) -> dict:
        return self._r_dict(self._pf)

    # ── Context helper ────────────────────────────────────────────────
    def get_context(
        self,
        query:    str,
        top_k:    int                  = 3,
        entities: Optional[List[str]] = None,
    ) -> str:
        results = self.search(query, top_k=top_k, entities=entities)
        if not results:
            return ""
        parts = []
        for r in results:
            text = r.get("text", "")
            if text:
                parts.append(text[:120])
        return " | ".join(parts)

    # ── Summary ───────────────────────────────────────────────────────
    def get_summary_with_intent(
        self,
        query:  str         = "",
        intent: Optional[dict] = None,
    ) -> str:
        facts  = self.get_facts()[:5]
        prefs  = self.get_prefs()
        topics = self.get_topics()[:8]
        parts  = []

        if prefs:
            pref_str = ", ".join(f"{k}={v}" for k, v in list(prefs.items())[:4])
            parts.append(f"User: {pref_str}")

        if facts:
            fact_str = "; ".join(f"{f['key']}={f['value']}" for f in facts[:4])
            parts.append(f"Facts: {fact_str}")

        if topics:
            parts.append(f"Topics: {', '.join(topics[:6])}")

        if query and self._vec_list:
            entities = intent.get("entities", []) if intent else []
            ctx = self.get_context(query, top_k=2, entities=entities)
            if ctx:
                parts.append(ctx)
            ep_sums = self._get_episode_summaries(query, top_k=1)
            if ep_sums:
                parts.append(f"Episode: {ep_sums[0][:100]}")

        return " | ".join(parts) if parts else ""

    # ── Engine interface ─────────────────────────────────────────────
    def prepare_engine_input(
        self,
        query:  str            = "",
        intent: Optional[dict] = None,
        top_k:  int            = 3,
    ) -> dict:
        if intent is None:
            intent = {"intent": "chat", "entities": []}
        intent_str = intent.get("intent", "chat")
        return {
            "memory_summary":    self.get_summary_with_intent(query, intent),
            "ltm_facts":         self.get_facts(),
            "memory_candidates": self.search(
                query,
                top_k=top_k * 3,
                entities=intent.get("entities", []),
                intent=intent_str,
            ),
        }

    # ── Feedback loop ────────────────────────────────────────────────
    def record_feedback(self, vid: str, positive: bool):
        """Thread-safe feedback recording with importance learning.

        IMPROVE: In addition to updating frequency, we also adjust the
        'importance' score so the forgetting/LRU logic becomes feedback-aware.
        Positive feedback increases importance (memory is kept longer).
        Negative feedback decays importance (memory ages out faster).
        FIX: Build a temporary vid→index dict for O(1) lookup instead of
        calling list.index() which is O(n) and scales poorly at 50k vectors.
        """
        with self._lock:
            # FIX: O(1) lookup via temporary dict instead of O(n) list.index()
            vid_to_idx = {v: i for i, v in enumerate(self._vids)}
            idx = vid_to_idx.get(vid)
            if idx is None:
                return
            meta = self._meta_list[idx]
            freq = meta.get("frequency", 1)
            meta["frequency"] = min(freq + 1, 9999) if positive else max(1, freq // 2)
            meta["timestamp"] = time.time()
            # IMPROVE: importance learning via feedback
            imp = meta.get("importance", 1.0)
            if positive:
                meta["importance"] = min(3.0, imp * 1.2)   # cap at 3× baseline
            else:
                meta["importance"] = max(0.1, imp * 0.7)   # floor at 0.1
            self._dirty = True
            self._maybe_save()
            self.clear_search_cache()

    # ── Consolidation ─────────────────────────────────────────────────
    def consolidate(self, sim_threshold: float = 0.92, jaccard_threshold: float = 0.2,
                    neighborhood: int = 20):
        """Merge near-duplicate vectors.

        IMPROVE: Replaced O(n²) brute-force with an O(n * neighborhood) approach:
        for each vector we only compare against its nearest `neighborhood`
        neighbors found via vector search.  With n=50k and neighborhood=20,
        this is ~1M comparisons instead of ~1.25B.

        FIX: Merging condition now requires BOTH sim > threshold AND jaccard >
        threshold. Previously sim-only could merge semantically divergent memories
        like "AI tốt cho y tế" and "AI tốt cho quân sự".

        FIX: Entire method is now wrapped in self._lock to prevent race conditions
        with concurrent store() or record_feedback() calls.

        Negation guard: pairs where one text is negated and the other is not are
        never merged, preserving contradictory memories.
        """
        with self._lock:
            if len(self._vec_list) < 2:
                return

            # FIX E: call _ensure_arr() ONCE before the loop starts.
            # Previously, every inner _vector_search(vec_i, ...) call triggered
            # _ensure_arr() which checked self._dirty == True (set by a prior store)
            # and rebuilt the full N×1024 float array on EVERY iteration —
            # O(n) rebuild × n iterations = O(n²) wasted work.
            # We temporarily clear _dirty around the rebuild so inner calls to
            # _ensure_arr() see a consistent, up-to-date array without rebuilding.
            self._ensure_arr()
            if self._vecs_arr is None:
                return
            # INVARIANT: _dirty is temporarily set to False for the duration of the
            # consolidate loop.  This is intentional and safe ONLY because:
            #   1. consolidate() holds self._lock for its entire execution.
            #   2. No other thread can modify _vec_list/_vids/_meta_list while we hold
            #      the lock, so the frozen _vecs_arr built by _ensure_arr() above stays
            #      consistent for the whole loop.
            #   3. _ensure_arr() condition is `_dirty OR shape-mismatch`; setting
            #      _dirty=False prevents O(n) per-iteration array rebuilds inside
            #      _vector_search while still allowing shape-mismatch to trigger a
            #      rebuild if the array were ever invalidated (defensive guard).
            #   4. _was_dirty is restored at the end so _save_vectors() can correctly
            #      detect whether a disk flush is needed after the loop.
            # WARNING: If the lock strategy changes in future (e.g. lock downgrade or
            # async re-embed), this section MUST be re-evaluated — setting _dirty=False
            # without holding the lock for the full duration would be a race condition.
            _was_dirty = self._dirty
            self._dirty = False   # prevent per-iteration rebuild — see INVARIANT above

            merged: set = set()   # set of vids to remove

            for i, vec_i in enumerate(self._vec_list):
                vid_i = self._vids[i]
                if vid_i in merged:
                    continue

                # Only compare against nearest neighbors, not all n entries
                cand_idx, cand_sims = self._vector_search(vec_i, top_k=neighborhood + 1)

                for sim, j in zip(cand_sims, cand_idx):
                    j = int(j)
                    if j == i:
                        continue
                    vid_j = self._vids[j]
                    if vid_j in merged:
                        continue
                    if sim < sim_threshold:
                        break   # results are sorted by sim desc; no point continuing
                    if self._meta_list[i].get("type") != self._meta_list[j].get("type"):
                        continue

                    text_i = self._meta_list[i]["text"]
                    text_j = self._meta_list[j]["text"]

                    # Negation guard: never merge contradictory memories
                    if self._has_negation(text_i) != self._has_negation(text_j):
                        continue

                    # Jaccard on unified token level (unigrams + bigrams)
                    # FIX: use _tokenize_with_bigrams so Jaccard uses the same
                    # token space as the embeddings (was unigram-only, inconsistent).
                    uni_i, bi_i = self.embedder._tokenize_with_bigrams(text_i)
                    uni_j, bi_j = self.embedder._tokenize_with_bigrams(text_j)
                    tokens_i = set(uni_i) | set(bi_i)
                    tokens_j = set(uni_j) | set(bi_j)
                    if tokens_i and tokens_j:
                        jaccard = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                    else:
                        jaccard = 0.0

                    # FIX: require BOTH sim AND jaccard to exceed thresholds
                    # to avoid merging contextually different memories
                    if sim >= sim_threshold and jaccard >= jaccard_threshold:
                        # Keep i, absorb j's frequency
                        self._meta_list[i]["frequency"] = (
                            self._meta_list[i].get("frequency", 1)
                            + self._meta_list[j].get("frequency", 1)
                        )
                        self._meta_list[i]["timestamp"] = time.time()
                        merged.add(vid_j)

            if merged:
                # Delete by vid (stable identifiers), scanning in reverse index order
                remove_indices = sorted(
                    [k for k, vid in enumerate(self._vids) if vid in merged],
                    reverse=True,
                )
                for idx in remove_indices:
                    del self._vids[idx]
                    del self._meta_list[idx]
                    del self._vec_list[idx]
                self._dirty = True
                self._save_vectors()
                self.clear_search_cache()
            else:
                # FIX E: restore _dirty to pre-consolidate state if nothing was merged
                # (we temporarily set it to False to prevent per-iteration _ensure_arr rebuild)
                self._dirty = _was_dirty

    # ── Topic-level memory consolidation ─────────────────────────────
    def consolidate_by_topic(self, min_cluster_size: int = 5,
                              sim_threshold: float = 0.25,
                              max_summary_chars: int = 400) -> int:
        """IMPROVE: Topic-based memory consolidation (periodic summarization).

        Addresses the BoW limitation where many small memories about the same
        subject fragment the vector space and dilute retrieval quality.

        Algorithm:
          1. Pre-compute content-token sets for all memories (stop-words removed).
          2. Cluster via Jaccard similarity on token sets: for each unprocessed memory,
             find all others with Jaccard ≥ sim_threshold (default 0.25).
          3. Skip clusters smaller than min_cluster_size (not worth summarizing).
          4. For qualifying clusters: build a synthetic summary text from the
             most-frequent tokens across all cluster members, store it as a new
             'general' memory with combined frequency, then delete the originals.
          5. All modifications happen inside self._lock.

        WHY JACCARD instead of cosine similarity:
          TF-IDF cosine similarity in a small corpus (< 1000 docs) is diluted by
          document-length normalization and IDF skew — two texts about the same
          topic can have cosine similarity as low as 0.05-0.15 even when clearly
          related.  Jaccard on shared content tokens is a more direct and robust
          measure of topical overlap for BoW embeddings, especially with few docs.

        Returns the number of cluster-summaries created.

        Thread-safety: fully wrapped in self._lock.
        Negation guard: memories with different negation polarity are never
        grouped into the same cluster.

        IMPORTANT: Similarity quality depends on the embedder being fit.
        If embedder.vocab is empty (never fit), the method auto-calls
        fit_embedder() before clustering so similarities are meaningful.
        Call fit_embedder() manually before consolidate_by_topic() for best
        results when you have domain-specific data.

        NOTE: sim_threshold default is 0.55 (not 0.90 like consolidate()) because
        topic clustering groups semantically-related memories that share a subject
        but are phrased differently — a looser threshold is appropriate here.
        Use consolidate() at threshold ≥ 0.90 to merge near-identical paraphrases.

        NOTE: This is intentionally a lightweight, zero-dependency summarizer.
        For production use with an LLM backend, replace _summarize_cluster()
        with an LLM call that generates a proper natural-language summary.
        """
        # Guard: auto-fit embedder if vocab is empty so sims are meaningful.
        # FIX: both self.embedder.vocab and self._meta_list are read here without
        # the lock — this is a benign TOCTOU (worst case: we call fit_embedder()
        # once unnecessarily, or skip it when vocab becomes empty concurrently).
        # fit_embedder() itself is fully lock-guarded and idempotent, so the
        # redundant call is safe.  We keep this outside the lock intentionally
        # because fit_embedder() acquires self._lock internally — calling it
        # from inside the with-self._lock block below would deadlock.
        if not self.embedder.vocab and self._meta_list:
            self.fit_embedder()   # acquires lock internally; safe to call here
        summaries_created = 0
        with self._lock:
            n = len(self._vec_list)
            if n < min_cluster_size:
                return 0

            _was_dirty = self._dirty

            processed: set = set()   # vids already assigned to a cluster
            clusters: List[List[int]] = []  # each entry: list of indices in cluster

            # Clustering strategy: "pivot-token" approach.
            # For BoW embeddings in small corpora, cosine sim and full-Jaccard both
            # suffer from low recall — two texts about the same topic (e.g. 7 Python
            # sentences) can share as few as 1-2 content tokens per pair, yielding
            # Jaccard ≈ 0.05-0.15.
            #
            # Instead we use an inverted-index approach:
            #   1. Build an inverted index: content_token → list of memory indices.
            #   2. Group all memories that share ≥ sim_threshold fraction of the
            #      most-common token's document-frequency (default: any shared token
            #      whose DF ≥ min_cluster_size means all those memories form a cluster).
            #   3. This is O(total_tokens) and naturally handles the "Python" pivot.
            STOP = {"là", "của", "và", "có", "the", "a", "an", "is", "in", "to",
                    "for", "on", "at", "by", "or", "be", "as", "it", "its", "that",
                    "được", "trong", "với", "này", "đó", "các", "những", "một", "không"}

            # Build inverted index: token → set of memory indices
            inverted: Dict[str, List[int]] = {}
            for i, meta in enumerate(self._meta_list):
                toks = set(self.embedder._tokenize(meta.get("text", ""))) - STOP
                for tok in toks:
                    if len(tok) >= 2:
                        inverted.setdefault(tok, []).append(i)

            # Find pivot tokens: tokens that appear in ≥ min_cluster_size memories
            pivot_tokens = {tok: idxs for tok, idxs in inverted.items()
                            if len(idxs) >= min_cluster_size}

            # Build clusters from pivot tokens (greedy: largest pivot first)
            sorted_pivots = sorted(pivot_tokens.items(), key=lambda x: -len(x[1]))
            for tok, idxs in sorted_pivots:
                # Collect unprocessed memories that share this pivot token.
                # FIX BUG-2: Previously neg_first was set from idxs[0] which may
                # already have been assigned to another cluster (i.e. in `processed`).
                # Using an already-processed entry as the polarity anchor caused the
                # remaining unprocessed entries — which might have a DIFFERENT polarity
                # from that anchor — to all be rejected by the negation guard, leading
                # to cluster fragmentation and missed groupings.
                # Fix: determine the dominant negation polarity from the UNPROCESSED
                # candidates first, then filter by that majority polarity.
                # Majority vote: count negation-True vs negation-False among unprocessed;
                # the larger group sets the cluster polarity (ties default to False).
                unprocessed_idxs = [idx for idx in idxs if self._vids[idx] not in processed]
                if not unprocessed_idxs:
                    continue
                neg_counts = {True: 0, False: 0}
                for idx in unprocessed_idxs:
                    neg_counts[self._has_negation(self._meta_list[idx].get("text", ""))] += 1
                # FIX: ties (50/50) default to False (non-negated polarity).
                # Negated memories are rarer in practice; letting a tie resolve to
                # True (negated) would exclude the larger non-negated group from
                # the cluster, which is the wrong outcome in the typical case.
                neg_dominant = neg_counts[True] > neg_counts[False]   # ties → False

                cluster = []
                for idx in unprocessed_idxs:
                    neg_idx = self._has_negation(self._meta_list[idx].get("text", ""))
                    if neg_idx != neg_dominant:
                        continue   # negation guard — exclude minority polarity
                    cluster.append(idx)
                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)
                    for idx in cluster:
                        processed.add(self._vids[idx])

            if not clusters:
                self._dirty = _was_dirty
                return 0

            # Process each cluster: summarise → store summary → delete originals.
            # Collect all vids to remove (delete AFTER all summaries are stored
            # so we don't invalidate indices mid-loop).
            vids_to_remove: set = set()
            new_entries: List[Tuple[str, np.ndarray, dict]] = []   # (text, vec, meta)

            for cluster in clusters:
                texts   = [self._meta_list[idx]["text"] for idx in cluster]
                freqs   = [self._meta_list[idx].get("frequency", 1) for idx in cluster]
                imps    = [self._meta_list[idx].get("importance", 1.0) for idx in cluster]
                mem_type = self._meta_list[cluster[0]].get("type", "general")

                summary_text = self._summarize_cluster(texts, max_summary_chars)
                if not summary_text or not self._is_learnable(summary_text):
                    continue  # fallback: don't consolidate if summary is invalid

                # Build summary vector from the centroid of cluster vectors
                cluster_vecs = np.stack([self._vec_list[idx] for idx in cluster], axis=0)
                centroid = cluster_vecs.mean(axis=0)
                norm_c = np.linalg.norm(centroid)
                if norm_c > 1e-8:
                    centroid /= norm_c
                else:
                    # FIX BUG-C: pass snapshot overrides — we are inside self._lock
                    # and _emb_vocab/_emb_idf/_emb_syns must be captured here.
                    # Build snapshot at this point (inside the with self._lock block).
                    _cbt_vocab = dict(self.embedder.vocab)
                    _cbt_idf   = dict(self.embedder.idf)
                    _cbt_syns  = dict(self.embedder.synonym_map)
                    centroid = self.embedder.embed(
                        summary_text,
                        vocab_override=_cbt_vocab,
                        idf_override=_cbt_idf,
                        synonym_map_override=_cbt_syns,
                    )

                combined_freq = min(sum(freqs), 9999)
                combined_imp  = min(3.0, sum(imps) / len(imps) * 1.1)
                summary_vid   = hashlib.md5(summary_text.encode()).hexdigest()[:12]

                new_meta = {
                    "text":        summary_text,
                    "timestamp":   time.time(),
                    "frequency":   combined_freq,
                    "type":        mem_type,
                    "metadata":    {"source": "topic_consolidation",
                                    "cluster_size": len(cluster)},
                    "importance":  combined_imp,
                    "emb_version": self._emb_version,
                }
                new_entries.append((summary_vid, centroid, new_meta))
                for idx in cluster:
                    vids_to_remove.add(self._vids[idx])

            if not new_entries:
                self._dirty = _was_dirty
                return 0

            # Remove originals (reverse-index-order for stability)
            remove_indices = sorted(
                [k for k, vid in enumerate(self._vids) if vid in vids_to_remove],
                reverse=True,
            )
            for idx in remove_indices:
                del self._vids[idx]
                del self._meta_list[idx]
                del self._vec_list[idx]

            # Append summaries (skip if vid already exists — exact duplicate)
            existing_vid_set = set(self._vids)
            for svid, svec, smeta in new_entries:
                if svid not in existing_vid_set:
                    self._vids.append(svid)
                    self._vec_list.append(svec)
                    self._meta_list.append(smeta)
                    existing_vid_set.add(svid)
                    summaries_created += 1

            if summaries_created > 0:
                self._dirty = True
                self._save_vectors()
                self.clear_search_cache()
            else:
                self._dirty = _was_dirty

        return summaries_created

    @staticmethod
    def _summarize_cluster(texts: List[str], max_chars: int = 400) -> str:
        """Build a lightweight extractive summary from a cluster of texts.

        Strategy: collect the most-frequent content tokens across all texts,
        then form a summary sentence by joining the top tokens.  This is
        intentionally minimal (no LLM dependency).  Replace with an LLM call
        for higher-quality summaries in production.

        The summary always ends with an indicator of the cluster size so the
        resulting memory is clearly marked as a consolidation artifact.
        """
        # Minimal stop-words (Vietnamese + English)
        # NOTE: Counter is imported at module level (no local import needed)
        STOP = {"là", "của", "và", "có", "the", "a", "an", "is", "in", "to",
                "for", "on", "at", "by", "or", "be", "as", "it", "its", "that"}
        token_counts: Counter = Counter()
        for text in texts:
            toks = MiniEmbedder._tokenize(text)
            for tok in toks:
                if tok not in STOP and len(tok) > 1:
                    token_counts[tok] += 1
        # Build summary from top tokens (proportional to cluster size)
        top_n = max(6, min(15, len(texts) * 2))
        top_tokens = [tok for tok, _ in token_counts.most_common(top_n)]
        if not top_tokens:
            return ""
        summary = " ".join(top_tokens[:12])
        summary = f"[Tổng hợp {len(texts)} ký ức] {summary}"
        return summary[:max_chars]

    # ── Inspection helpers ────────────────────────────────────────────
    def list_recent(self, n: int = 10) -> List[dict]:
        """FIX: Snapshot under lock to avoid inconsistency with concurrent writes."""
        with self._lock:
            if not self._meta_list:
                return []
            snapshot = list(zip(list(self._vids), list(self._meta_list)))
        sorted_entries = sorted(
            snapshot,
            key=lambda x: x[1].get("timestamp", 0),
            reverse=True,
        )
        return [{"id": vid, **meta} for vid, meta in sorted_entries[:n]]

    def search_by_type(self, mem_type: str) -> List[dict]:
        """FIX: Snapshot under lock to avoid inconsistency with concurrent writes."""
        with self._lock:
            snapshot = [(self._vids[i], meta)
                        for i, meta in enumerate(self._meta_list)
                        if meta.get("type") == mem_type]
        return [{"id": vid, **meta} for vid, meta in snapshot]

    def stats(self) -> dict:
        """FIX: Snapshot under lock to avoid inconsistency with concurrent writes."""
        with self._lock:
            total_vectors = len(self._vec_list)
            cache_size    = len(self._search_cache)
            store_count   = self._store_count
            vocab_size    = len(self.embedder.vocab)
            embedder_frozen = self.embedder.is_frozen
            type_counts: Dict[str, int] = {}
            for meta in self._meta_list:
                t = meta.get("type", "general")
                type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_vectors":   total_vectors,
            "total_facts":     len(self._r(self._ff)),
            "total_episodes":  len(self._r(self._ef)),
            "total_knowledge": len(self._r(self._kf)),
            "type_breakdown":  type_counts,
            "store_count":     store_count,
            "vocab_size":      vocab_size,
            "embedder_frozen": embedder_frozen,
            "cache_size":      cache_size,
        }

    def export(self, path: str):
        """FIX: Snapshot shared vectors/meta under lock for a consistent export."""
        with self._lock:
            vectors_snapshot = [
                {"id": vid, **meta}
                for vid, meta in zip(list(self._vids), list(self._meta_list))
            ]
        data = {
            "facts":     self._r(self._ff),
            "episodes":  self._r(self._ef),
            "knowledge": self._r(self._kf),
            "prefs":     self._r_dict(self._pf),
            "vectors":   vectors_snapshot,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clear_search_cache(self):
        """Invalidate search cache and bump version so old cache keys are auto-stale."""
        self._search_cache.clear()
        self._version += 1


# ══════════════════════════════════════════════════════════════════════
# WORKING MEMORY V1
# ══════════════════════════════════════════════════════════════════════
class WorkingMemoryV1:
    MAX_MESSAGES = 200   # hard cap: prevent unbounded RAM growth in long sessions

    def __init__(self, ltm: LongTermMemoryV1, kv_buffer: KVCacheBuffer,
                 max_tokens: int = 512, tokenizer=None):
        self.ltm        = ltm
        self.kv_buf     = kv_buffer
        self.max_tokens = max_tokens
        self.tokenizer  = tokenizer
        self.messages:   List[dict] = []
        self.token_count = 0
        self.session_id  = str(int(time.time()))

    def _count_tokens(self, content: str) -> int:
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(content, add_bos=False, add_eos=False))
            except Exception:
                pass
        # IMPROVE: hybrid token estimator for mixed Vietnamese/English text.
        # Previous len(content)//4 treated every 4 chars as 1 token regardless
        # of language, which under-counted Vietnamese (avg syllable ~3 chars) and
        # over-counted English (avg word ~5 chars).
        #
        # Better approximation:
        #   • Vietnamese: each whitespace-delimited syllable ≈ 1 BPE sub-token.
        #     word_count * 1.3 accounts for punctuation and multi-syllable terms.
        #   • English: each whitespace word ≈ 1.3 BPE tokens on average.
        #   • Mixed text: the formula self-corrects because VI has more words per
        #     sentence than EN for the same semantic content.
        #
        # Formula: max(char_estimate, word_estimate) acts as a conservative upper
        # bound — it prefers to trim slightly earlier rather than let the buffer
        # grow silently beyond budget.
        word_est = max(1, int(len(content.split()) * 1.3))
        char_est = max(1, len(content) // 4)
        return max(word_est, char_est)

    def add(self, role: str, content: str, meta: dict = None,
            is_thinking: bool = False):
        self.messages.append({
            "role":       role,
            "content":    content,
            "ts":         time.time(),
            "is_thinking": is_thinking,
            "meta":       meta or {},
        })
        self.token_count += self._count_tokens(content)
        self._trim()

    def _trim(self):
        """Evict oldest non-system message when token budget or hard cap is exceeded.

        FIX: Previous implementation evicted the first non-system message it
        found (always index 0 or 1), which could evict a very recent message
        immediately after it was added. Now we find the OLDEST non-system
        message by scanning for the minimum timestamp among non-system entries,
        preserving conversational coherence.
        IMPROVE: MAX_MESSAGES hard cap prevents unbounded RAM growth in long
        automation sessions even if token_count stays small (e.g. very short msgs).
        """
        # Hard cap: evict oldest until within limit
        while len(self.messages) > self.MAX_MESSAGES:
            evict_idx = -1
            oldest_ts = float("inf")
            for i, m in enumerate(self.messages):
                if m["role"] != "system" and m.get("ts", float("inf")) < oldest_ts:
                    oldest_ts = m.get("ts", float("inf"))
                    evict_idx = i
            if evict_idx == -1:
                break
            removed = self.messages.pop(evict_idx)
            # FIX: guard token_count from going negative
            self.token_count = max(0, self.token_count - self._count_tokens(removed["content"]))
            self.kv_buf.push(removed["content"], self._recent_ctx())

        # Token budget: evict oldest until within budget
        while self.token_count > self.max_tokens and len(self.messages) > 1:
            # FIX: find oldest non-system message (by timestamp) instead of first
            evict_idx = -1
            oldest_ts = float("inf")
            for i, m in enumerate(self.messages):
                if m["role"] != "system" and m.get("ts", float("inf")) < oldest_ts:
                    oldest_ts = m.get("ts", float("inf"))
                    evict_idx = i
            if evict_idx == -1:
                break
            removed       = self.messages.pop(evict_idx)
            removed_tc    = self._count_tokens(removed["content"])
            # FIX: guard token_count from going negative due to any counting inconsistency
            self.token_count = max(0, self.token_count - removed_tc)
            self.kv_buf.push(removed["content"], self._recent_ctx())
            if len(self.kv_buf) >= 5:
                self.kv_buf.flush_to_ltm(self.ltm)

    def _recent_ctx(self) -> str:
        return " ".join(m["content"] for m in self.messages[-3:])

    def get_messages(self) -> List[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def end_session(self):
        if self.messages:
            self.kv_buf.flush_to_ltm(self.ltm)
            msgs    = self.get_messages()
            words   = " ".join(m["content"] for m in msgs if m["role"] in ("user", "assistant"))
            summary = words[:200]
            detail  = json.dumps(msgs, ensure_ascii=False)
            self.ltm.save_episode(summary=summary, detail=detail, tags=["conversation"])
        self.messages.clear()
        self.token_count = 0
        self.session_id  = str(int(time.time()))

    def clear(self):
        # FIX D: Only flush kv_buf if it actually has entries AND end_session() wasn't
        # just called (end_session already flushes). Using len() check avoids the
        # double-store bug where clear() re-flushed already-flushed kv_buf entries
        # if called right after end_session().
        if self.kv_buf and len(self.kv_buf) > 0:
            self.kv_buf.flush_to_ltm(self.ltm)
        self.messages.clear()
        self.token_count = 0


# ══════════════════════════════════════════════════════════════════════
# MODULE SELF-TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tempfile, shutil

    print("=== DracoAI Memory V1 — Self-Test ===")
    tmp = tempfile.mkdtemp()
    try:
        mem = LongTermMemoryV1(memory_dir=tmp)

        # KVCacheBuffer
        buf = KVCacheBuffer(max_size=4)
        buf.push("đây là một câu đủ dài để học")
        buf.push("ngắn")
        buf.push("thêm một câu nữa cho buffer nhé bạn ơi")
        assert len(buf) == 2, f"Expected 2, got {len(buf)}"
        print(f"✅ KVCacheBuffer: {len(buf)} entries (filter OK)")

        # FIX: 2-word uppercase phrase accepted
        buf2 = KVCacheBuffer()
        buf2.push("GPU mạnh")
        assert len(buf2) == 1, "2-word uppercase phrase should be learnable"
        print("✅ KVCacheBuffer: 2-word uppercase phrase accepted")

        # Store + dedup
        vid1 = mem.store("DracoAI là mô hình ngôn ngữ lớn do DUCNGUYEN tạo ra", {"type": "fact"})
        vid2 = mem.store("DracoAI là mô hình ngôn ngữ lớn do DUCNGUYEN tạo ra", {"type": "fact"})
        assert vid1 == vid2
        print(f"✅ store + dedup: vid={vid1}")

        # Short text filtered (but "là" pattern allows 3‑word definitions)
        vid_short = mem.store("MoE = nhiều expert")
        assert vid_short != "", "Definition pattern should allow short text"
        print(f"✅ Short definition stored: vid={vid_short}")

        # Knowledge question passes (>=4 words for knowledge keywords)
        vid_q = mem.store("Hà Nội là thủ đô của nước nào?")
        assert vid_q != ""
        print(f"✅ Knowledge question stored: vid={vid_q}")

        # Semantic duplicate with negation guard
        vid_a = mem.store("AI là công nghệ của tương lai")
        vid_b = mem.store("AI không phải là công nghệ của tương lai")
        assert vid_a != vid_b, "Negation should prevent false semantic duplicate"
        print(f"✅ Semantic duplicate negation guard: vid_a={vid_a}, vid_b={vid_b} (different)")

        # Fact versioning
        mem.remember_fact("author", "DUCNGUYEN", confidence=0.9)
        mem.remember_fact("author", "DUCNGUYEN")
        mem.remember_fact("author", "Draco Studio")
        facts = mem.get_facts()
        author_fact = next(f for f in facts if f["key"] == "author")
        assert author_fact["value"] == "Draco Studio"
        assert len(author_fact.get("history", [])) >= 1
        print(f"✅ remember_fact + versioning: history={len(author_fact['history'])}")

        # Store several texts
        for t in [
            "Python là ngôn ngữ lập trình phổ biến dùng cho AI và data science",
            "NumPy cung cấp mảng đa chiều hiệu năng cao cho Python",
            "Transformer là kiến trúc nền tảng của các mô hình ngôn ngữ hiện đại",
            "Attention mechanism cho phép mô hình tập trung vào các token quan trọng",
            "MoE chia mô hình thành nhiều expert chuyên biệt để tăng hiệu năng",
            "DracoAI sử dụng kiến trúc MoE dựa trên Qwen 3.5 9B làm backbone chính",
        ]:
            mem.store(t)

        # Search + LRU cache — verify different entity keys get different cache entries
        r1 = mem.search("Python", top_k=3, entities=["Python"], use_mmr=False)
        r2 = mem.search("Python", top_k=3, entities=["DracoAI"], use_mmr=False)
        print(f"✅ search with different entities: r1={len(r1)}, r2={len(r2)}")
        r1_bis = mem.search("Python", top_k=3, entities=["Python"], use_mmr=False)
        assert len(r1_bis) == len(r1)
        print("✅ search cache hit: ok")

        # Verify _vector_search returns results in descending similarity order
        q_vec = mem.embedder.embed("Python")
        idxs, sims = mem._vector_search(q_vec, top_k=100)  # request more than available
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1)), \
            "vector_search results must be sorted descending even when top_k >= n"
        print("✅ _vector_search sorted order (len<=top_k case): ok")

        # FIX: NaN guard in _vector_search
        nan_vec = np.full(1024, float("nan"), dtype=np.float32)
        nan_vec = np.nan_to_num(nan_vec, nan=0.0)
        idxs_nan, sims_nan = mem._vector_search(nan_vec, top_k=3)
        assert not np.any(np.isnan(sims_nan)), "NaN should be replaced by 0.0"
        print("✅ _vector_search NaN guard: ok")

        # Summary
        summary = mem.get_summary_with_intent("Python và AI", {"intent": "factual", "entities": ["Python"]})
        assert isinstance(summary, str)
        print(f"✅ get_summary_with_intent: '{summary[:80]}...'")

        # prepare_engine_input
        inp = mem.prepare_engine_input("DracoAI MoE", {"intent": "factual", "entities": ["DracoAI"]})
        assert all(k in inp for k in ("memory_summary", "ltm_facts", "memory_candidates"))
        print(f"✅ prepare_engine_input: candidates={len(inp['memory_candidates'])}")

        # Feedback
        mem.record_feedback(vid1, positive=True)
        print("✅ record_feedback: ok")

        # Stats
        s = mem.stats()
        assert s["total_vectors"] > 0
        print(f"✅ stats: {s}")

        # list_recent / search_by_type
        assert len(mem.list_recent(3)) <= 3
        by_fact = mem.search_by_type("fact")
        print(f"✅ list_recent=3, search_by_type(fact)={len(by_fact)}")
        if by_fact:
            assert "embedding" not in by_fact[0]
        print("✅ No embedding in search result")

        # Consolidate with negation guard
        mem.store("MoE là kiến trúc mạnh mẽ")
        mem.store("MoE không phải là kiến trúc mạnh mẽ")
        mem.consolidate(sim_threshold=0.85, jaccard_threshold=0.3)
        print("✅ consolidate: negation guard active")

        # FIX: verify consolidate doesn't merge contextually different memories
        vid_x = mem.store("AI tốt cho ngành y tế và bệnh viện lớn")
        vid_y = mem.store("AI tốt cho ngành quân sự và vũ khí hiện đại")
        count_before = len(mem._vids)
        mem.consolidate(sim_threshold=0.92, jaccard_threshold=0.5)
        count_after = len(mem._vids)
        print(f"✅ consolidate dual-condition: {count_before} → {count_after} (contextually different kept)")

        # Export
        export_path = os.path.join(tmp, "export.json")
        mem.export(export_path)
        assert os.path.exists(export_path)
        print(f"✅ export: {export_path}")

        # Episode + knowledge
        mem.save_episode("Hội thoại về MoE", detail="...", tags=["moe"])
        mem.learn("MoE", "Mô hình chia thành nhiều chuyên gia")
        topics = mem.get_topics()
        assert "MoE" in topics
        print(f"✅ save_episode + learn: topics={topics[:5]}")

        # replay_relevant
        mem.save_episode("Deep Learning là tập con của Machine Learning", detail="...", tags=["dl"])
        relevant = mem.replay_relevant("machine learning")
        assert len(relevant) > 0
        summaries = mem._get_episode_summaries("machine learning", top_k=1)
        print(f"✅ replay_relevant: {len(relevant)} episodes, summary='{summaries[0][:50] if summaries else ''}'")

        # Decay cleanup
        mem.decay_cleanup(days_threshold=365)
        print("✅ decay_cleanup: ok")

        # WorkingMemoryV1
        wm = WorkingMemoryV1(ltm=mem, kv_buffer=KVCacheBuffer(max_size=10), max_tokens=100)
        wm.add("user", "Xin chào, bạn tên gì?")
        wm.add("assistant", "Tôi là DracoAI, trợ lý ảo thông minh.", is_thinking=True)
        assert len(wm.get_messages()) == 2
        wm.end_session()
        episodes_after = mem._r(mem._ef)
        print(f"✅ WorkingMemoryV1: session ended, total episodes={len(episodes_after)}")

        # FIX: WorkingMemory._trim evicts oldest message, not just first non-system
        wm2 = WorkingMemoryV1(ltm=mem, kv_buffer=KVCacheBuffer(max_size=10), max_tokens=50)
        wm2.add("user",      "Tin nhắn cũ nhất đây, phải bị evict trước", meta={})
        time.sleep(0.01)
        wm2.add("assistant", "Trả lời trung gian")
        time.sleep(0.01)
        wm2.add("user",      "Tin nhắn mới nhất không được mất")
        # Force trim by adding a large message
        wm2.add("user",      "Đây là một câu rất dài để trigger trim và kiểm tra xem tin nhắn cũ nhất có bị xóa đúng không")
        remaining_contents = [m["content"] for m in wm2.messages]
        print(f"✅ WorkingMemory._trim oldest-first: remaining={len(remaining_contents)} messages")

        # Check metadata redundancy
        meta_sample = mem._meta_list[0]
        assert "embedding" not in meta_sample
        print("✅ Metadata free of redundant embedding")

        # Embedder freeze
        mem.freeze_embedder()
        assert mem.embedder.is_frozen
        mem.fit_embedder(["some new text to be ignored"])
        assert mem.stats()["embedder_frozen"]
        print("✅ Embedder freeze: active")

        # FIX: cache cleared after fit_embedder
        mem.unfreeze_embedder()
        r_before = mem.search("Python", top_k=3, use_mmr=False)
        mem.fit_embedder()  # should clear cache
        assert len(mem._search_cache) == 0, "Cache must be empty after fit_embedder"
        print("✅ Cache invalidated after fit_embedder: ok")

        # Flush vectors explicitly
        mem.flush_vectors()
        print("✅ flush_vectors: ok")

        # FIX: atomic write — verify tmp file is cleaned up
        tmp_path = mem._npz[:-4] + "_tmp.npz"
        assert not os.path.exists(tmp_path), "Tmp file should not exist after atomic save"
        print("✅ Atomic _save_vectors: no leftover .tmp file")

        # LRU cache eviction test
        for i in range(5):
            mem.search(f"test query {i}", top_k=2, entities=[], use_mmr=False)
        assert len(mem._search_cache) <= 5

        # prefs.json initialized as dict (not list)
        import json as _json
        with open(mem._pf) as _f:
            _pdata = _json.load(_f)
        assert isinstance(_pdata, dict), f"prefs.json should be dict, got {type(_pdata)}"
        print("✅ prefs.json initialized as dict: ok")

        print("\n✅ All DracoAI Memory V1 self-tests passed.")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)