# DracoAI V1 — tokenizer/bpe/trainer.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
DracoAI Tokenizer — BPE Trainer
================================
Trains BPE merge rules from a text corpus.
Reference implementation; for large corpora use the HuggingFace
``tokenizers`` library and import via compatibility/hf_import.py.
"""

from collections import Counter
from typing import Iterator, List, Tuple


def _corpus_to_words(text_iter: Iterator[str]) -> Counter:
    counts: Counter = Counter()
    for text in text_iter:
        for word in text.split():
            if word:
                counts[tuple(word.encode("utf-8"))] += 1
    return counts


def _count_pairs(word_counts: Counter) -> Counter:
    pairs: Counter = Counter()
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += count
    return pairs


def _apply_merge(pair: Tuple[int, int], new_id: int,
                 word_counts: Counter) -> Counter:
    a, b = pair
    new_counts: Counter = Counter()
    for word, count in word_counts.items():
        new_word: List[int] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_counts[tuple(new_word)] += count
    return new_counts


def train_bpe(
    text_iter: Iterator[str],
    target_vocab_size: int = 32000,
    base_vocab_size:   int = 256,
    verbose:           bool = False,
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Train BPE merges from *text_iter*.

    Returns
    -------
    List[Tuple[Tuple[int,int], int]]
        Merge rules: ((left_id, right_id), merged_id).
        Pass to MergeEngine.load().
    """
    word_counts = _corpus_to_words(text_iter)
    next_id     = base_vocab_size
    merges: List[Tuple[Tuple[int, int], int]] = []
    num_merges  = target_vocab_size - base_vocab_size

    for step in range(num_merges):
        pair_counts = _count_pairs(word_counts)
        if not pair_counts:
            break
        best_pair  = max(pair_counts, key=lambda p: pair_counts[p])
        best_count = pair_counts[best_pair]
        if best_count < 2:
            break

        merged_id   = next_id
        next_id    += 1
        merges.append((best_pair, merged_id))
        word_counts = _apply_merge(best_pair, merged_id, word_counts)

        if verbose and (step + 1) % 100 == 0:
            a, b = best_pair
            print(f"  merge {step+1}/{num_merges}: ({a}, {b})→{merged_id}  freq={best_count}")

    return merges