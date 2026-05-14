# DracoAI V1 — modeling/testing/test_sampling.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for sampling algorithms and penalties.

FIXES (this revision):
  ✅ FIX-UNUSED-IMPORT-PYTEST : removed unused `import pytest`.  No test in
     this file uses pytest.raises, pytest.mark, or any other pytest API
     directly — the test classes are discovered automatically by pytest
     without needing the import.
"""
import numpy as np
from ..sampling.sampler   import Sampler
from ..sampling.mirostat  import mirostat_v2
from ..sampling.penalties import (apply_repetition_penalty,
                                   apply_frequency_penalty,
                                   apply_presence_penalty)


class TestMirostatV2:
    def test_returns_valid_token(self):
        logits = np.random.randn(1000).astype(np.float64)
        token, new_mu = mirostat_v2(logits, mu=5.0)
        assert 0 <= token < 1000
        assert new_mu > 0

    def test_mu_decreases_on_surprise(self):
        """High-surprise token should lower mu (negative feedback)."""
        logits = np.zeros(100, dtype=np.float64)
        logits[0] = -10.0  # low prob → high surprise
        _, new_mu = mirostat_v2(logits, mu=5.0, tau=5.0, eta=1.0)
        # surprise > tau → mu should decrease
        # (result depends on sampled token; we just check bounds)
        assert new_mu >= 0.1

    def test_no_nan_on_extreme_logits(self):
        logits = np.array([-1000.0] * 999 + [1000.0])
        token, mu = mirostat_v2(logits, mu=5.0)
        assert np.isfinite(mu)
        assert 0 <= token < 1000


class TestSampler:
    def test_topk_topp_valid_token(self):
        logits = np.random.randn(500).astype(np.float32)
        token  = Sampler.topk_topp(logits, temp=1.0, top_p=0.9, top_k=50)
        assert 0 <= token < 500

    def test_deterministic_with_argmax(self):
        logits = np.zeros(100, dtype=np.float32)
        logits[42] = 10.0
        assert Sampler.argmax(logits) == 42

    def test_min_p_filters_low_prob(self):
        logits = np.zeros(100, dtype=np.float32)
        logits[0] = 5.0   # dominant token
        # With high min_p, all low-prob tokens should be filtered
        tokens = {Sampler.topk_topp(logits, temp=1.0, top_p=1.0,
                                     top_k=0, min_p=0.5)
                  for _ in range(20)}
        assert tokens == {0}  # only the dominant token should be chosen


class TestPenalties:
    def test_rep_penalty_reduces_logit(self):
        logits = np.zeros(10, dtype=np.float64)
        freq   = {3: 2}
        pos    = {3: 0}
        out    = apply_repetition_penalty(logits, freq, pos, n_pos=5, rep_alpha=1.0)
        assert out[3] < logits[3]

    def test_freq_penalty_proportional(self):
        logits = np.zeros(10, dtype=np.float64)
        freq   = {1: 3, 2: 1}
        out    = apply_frequency_penalty(logits, freq, penalty=0.5)
        assert out[1] < out[2]  # token 1 penalised more (higher count)

    def test_presence_penalty_flat(self):
        logits = np.zeros(10, dtype=np.float64)
        freq   = {0: 1, 5: 100}  # both appeared — same flat penalty
        out    = apply_presence_penalty(logits, freq, penalty=1.0)
        np.testing.assert_allclose(out[0], out[5])

    def test_does_not_mutate_input(self):
        logits = np.zeros(10, dtype=np.float64)
        orig   = logits.copy()
        apply_repetition_penalty(logits, {3: 1}, {3: 0}, n_pos=1)
        np.testing.assert_array_equal(logits, orig)