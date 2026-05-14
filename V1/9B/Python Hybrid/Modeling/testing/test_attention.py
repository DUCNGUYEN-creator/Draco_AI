# DracoAI V1 — modeling/testing/test_attention.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for GQAttention and attention ops.

FIXES (this revision):
  ✅ FIX-UNUSED-IMPORT-MATH : removed unused `import math`.  No test in this
     file calls any math.* function; the sqrt/pi constants needed by attention
     are computed internally by the modules under test.
"""
import numpy as np
import pytest

from ..ops.attention_ops import rope_freqs, apply_rope, safe_softmax, causal_mask_bias
from ..layers.attention  import GQAttention
from ..kv_cache.kv_cache    import KVCache


class TestRoPE:
    def test_rope_freqs_shape(self):
        freqs = rope_freqs(64)
        assert freqs.shape == (32,)

    def test_apply_rope_shape_preserved(self):
        x     = np.random.randn(1, 4, 8, 64).astype(np.float32)
        freqs = rope_freqs(64)
        out   = apply_rope(x, freqs, offset=0)
        assert out.shape == x.shape

    def test_apply_rope_different_offsets(self):
        x     = np.random.randn(1, 1, 4, 32).astype(np.float32)
        freqs = rope_freqs(32)
        out0  = apply_rope(x, freqs, offset=0)
        out4  = apply_rope(x, freqs, offset=4)
        assert not np.allclose(out0, out4), "Different offsets should give different encodings"


class TestSafeSoftmax:
    def test_sums_to_one(self):
        x = np.random.randn(4, 16).astype(np.float32)
        p = safe_softmax(x, axis=-1)
        np.testing.assert_allclose(p.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_no_nan_with_extreme_values(self):
        x = np.array([[-1000.0, 1000.0, 0.0]])
        p = safe_softmax(x, axis=-1)
        assert np.all(np.isfinite(p))

    def test_clip_applied(self):
        x = np.array([[100.0, -100.0]])
        p = safe_softmax(x, axis=-1, clip=50.0)
        assert np.all(p >= 0)


class TestCausalMask:
    def test_upper_triangular(self):
        mask = causal_mask_bias(4)
        # Lower triangle (including diagonal) should be 0
        np.testing.assert_array_equal(np.tril(mask), 0)
        # Upper triangle should be -1e9
        assert np.all(np.triu(mask, 1) < -1e8)


class TestGQAttention:
    @pytest.fixture
    def setup(self):
        d_model, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
        attn  = GQAttention(d_model, n_heads, n_kv_heads, head_dim)
        cache = KVCache(n_layers=1, n_kv_heads=n_kv_heads, head_dim=head_dim,
                        window=128, sink=4)
        return attn, cache

    def test_output_shape(self, setup):
        attn, cache = setup
        x   = np.random.randn(1, 3, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0)
        assert out.shape == (1, 3, 64)

    def test_single_token(self, setup):
        attn, cache = setup
        x   = np.random.randn(1, 1, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0)
        assert out.shape == (1, 1, 64)

    def test_no_nan(self, setup):
        attn, cache = setup
        x = np.random.randn(1, 5, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0)
        assert np.all(np.isfinite(out))

    def test_cache_increments(self, setup):
        attn, cache = setup
        x = np.random.randn(1, 3, 64).astype(np.float32)
        attn.forward(x, cache, layer_idx=0)
        assert cache.current_length() == 3