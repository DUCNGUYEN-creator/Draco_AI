# DracoAI V1 — modeling/testing/test_attention.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for GQAttention, attention ops, MLA, and hybrid attention.

FIXES retained:
  ✅ FIX-UNUSED-IMPORT-MATH : removed unused import math.

NEW tests in this revision:
  • TestMLAProjection         — compress/expand, shapes, ratio
  • TestHybridAttentionConfig — layer classification, best_engram_layer
  • TestGQAttentionMLA        — GQAttention with MLA attached
  • TestTokenSparsity         — sparsity_thresh effect on prefill attention
"""
import numpy as np
import pytest

from ..ops.attention_ops      import rope_freqs, apply_rope, safe_softmax, causal_mask_bias
from ..layers.attention       import GQAttention
from ..layers.attention_mla   import MLAProjection
from ..layers.hybrid_attention import HybridAttentionConfig, build_default_global_layers
from ..kv_cache.kv_cache      import KVCache


# ─────────────────────────────────────────────────────────────────────────────
# Original tests (all retained)
# ─────────────────────────────────────────────────────────────────────────────

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
        assert not np.allclose(out0, out4)


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
        # Lower triangle (including diagonal) must be 0
        rows_l, cols_l = np.tril_indices(4)
        np.testing.assert_array_equal(mask[rows_l, cols_l], 0)
        # Strictly upper triangle must be large-negative
        rows_u, cols_u = np.triu_indices(4, k=1)
        assert np.all(mask[rows_u, cols_u] < -1e8)


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


# ─────────────────────────────────────────────────────────────────────────────
# NEW tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMLAProjection:
    """Multi-head Latent Attention projection tests."""

    def test_init_shapes(self):
        mla = MLAProjection(n_kv_heads=4, head_dim=32, latent_dim=8)
        assert mla.W_kc.shape == (32, 8)
        assert mla.W_ke.shape == (8,  32)
        assert mla.W_vc.shape == (32, 8)
        assert mla.W_ve.shape == (8,  32)

    def test_compression_shape(self):
        mla = MLAProjection(n_kv_heads=2, head_dim=16, latent_dim=4)
        K = np.random.randn(1, 2, 8, 16).astype(np.float32)
        K_lat = mla.compress_k(K)
        assert K_lat.shape == (1, 2, 8, 4)

    def test_expansion_shape(self):
        mla = MLAProjection(n_kv_heads=2, head_dim=16, latent_dim=4)
        K_lat = np.random.randn(1, 2, 8, 4).astype(np.float32)
        K_exp = mla.expand_k(K_lat)
        assert K_exp.shape == (1, 2, 8, 16)

    def test_compression_ratio(self):
        mla = MLAProjection(n_kv_heads=4, head_dim=64, latent_dim=16)
        assert abs(mla.compression_ratio() - 0.25) < 1e-6

    def test_orthonormal_init_low_error(self):
        """With orthonormal W_kc, compress→expand has minimal subspace loss."""
        mla = MLAProjection(n_kv_heads=1, head_dim=16, latent_dim=8)
        # W_kc is initialised as orthonormal columns → W_ke = W_kc.T
        # The top-8 singular vectors are perfectly preserved.
        # Create a K that lies exactly in the column space of W_kc
        K_in_space = (np.random.randn(1, 1, 4, 8).astype(np.float32)
                      @ mla.W_kc.T)  # (1, 1, 4, 16)
        K_lat  = mla.compress_k(K_in_space)
        K_back = mla.expand_k(K_lat)
        np.testing.assert_allclose(K_in_space, K_back, atol=1e-5)

    def test_invalid_latent_dim_raises(self):
        with pytest.raises(ValueError):
            MLAProjection(n_kv_heads=2, head_dim=16, latent_dim=16)

    def test_v_compress_expand_shapes(self):
        mla = MLAProjection(n_kv_heads=2, head_dim=32, latent_dim=8)
        V = np.random.randn(1, 2, 5, 32).astype(np.float32)
        V_lat  = mla.compress_v(V)
        V_back = mla.expand_v(V_lat)
        assert V_lat.shape  == (1, 2, 5, 8)
        assert V_back.shape == (1, 2, 5, 32)
        assert np.all(np.isfinite(V_back))


class TestHybridAttentionConfig:
    """HybridAttentionConfig layer classification tests."""

    def test_default_global_layers_small(self):
        cfg = HybridAttentionConfig(n_layers=4)
        assert 0 in cfg.global_layers
        assert (cfg.n_layers - 1) in cfg.global_layers

    def test_default_global_all_small(self):
        cfg = HybridAttentionConfig(n_layers=2)
        assert cfg.global_layers == [0, 1]

    def test_is_global_is_local_exclusive(self):
        cfg = HybridAttentionConfig(n_layers=6)
        for i in range(6):
            assert cfg.is_global(i) != cfg.is_local(i)

    def test_custom_global_layers(self):
        cfg = HybridAttentionConfig(n_layers=8, global_layers=[1, 4, 6])
        assert cfg.is_global(1) and cfg.is_global(4) and cfg.is_global(6)
        assert cfg.is_local(0) and cfg.is_local(2) and cfg.is_local(3)

    def test_out_of_range_ignored(self):
        cfg = HybridAttentionConfig(n_layers=4, global_layers=[0, 99, -1])
        assert 0 in cfg.global_layers
        assert 99 not in cfg.global_layers
        assert -1 not in cfg.global_layers

    def test_global_kv_limit_unlimited(self):
        cfg = HybridAttentionConfig(n_layers=4, global_window=0)
        assert cfg.global_kv_limit(cfg.global_layers[0], 1000) == 1000

    def test_global_kv_limit_bounded(self):
        cfg = HybridAttentionConfig(n_layers=4, global_window=512)
        assert cfg.global_kv_limit(cfg.global_layers[0], 2000) == 512
        assert cfg.global_kv_limit(cfg.global_layers[0], 128)  == 128

    def test_best_engram_layer(self):
        cfg = HybridAttentionConfig(n_layers=6)
        best = cfg.best_engram_layer()
        assert best in cfg.global_layers

    def test_build_default_global_layers(self):
        layers = build_default_global_layers(8)
        assert 0 in layers and 7 in layers
        assert len(layers) >= 2

    def test_summary_keys(self):
        cfg = HybridAttentionConfig(n_layers=4)
        s = cfg.summary()
        assert "global_layers" in s and "local_layers" in s
        assert len(s["global_layers"]) + len(s["local_layers"]) == 4


class TestGQAttentionMLA:
    """GQAttention with MLAProjection (compressed KV storage)."""

    @pytest.fixture
    def setup_mla(self):
        d_model, n_heads, n_kv_heads, head_dim = 64, 4, 2, 16
        attn  = GQAttention(d_model, n_heads, n_kv_heads, head_dim)
        cache = KVCache(n_layers=1, n_kv_heads=n_kv_heads, head_dim=8,
                        window=64, sink=4)  # head_dim=8 for latent storage
        mla   = MLAProjection(n_kv_heads=n_kv_heads, head_dim=head_dim, latent_dim=8)
        return attn, cache, mla

    def test_mla_output_shape(self, setup_mla):
        attn, cache, mla = setup_mla
        x   = np.random.randn(1, 3, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0, mla=mla)
        assert out.shape == (1, 3, 64)

    def test_mla_no_nan(self, setup_mla):
        attn, cache, mla = setup_mla
        x   = np.random.randn(1, 5, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0, mla=mla)
        assert np.all(np.isfinite(out))

    def test_mla_cache_smaller_than_full(self, setup_mla):
        """With MLA, cache stores latent_dim vectors, not full head_dim."""
        _, cache, mla = setup_mla
        assert cache.head_dim == mla.latent_dim  # 8, not 16

    def test_mla_multi_step(self, setup_mla):
        """Multiple decode steps with MLA produce consistent output shapes."""
        attn, cache, mla = setup_mla
        for _ in range(5):
            x = np.random.randn(1, 1, 64).astype(np.float32)
            out = attn.forward(x, cache, layer_idx=0, mla=mla)
            assert out.shape == (1, 1, 64)
            assert np.all(np.isfinite(out))


class TestTokenSparsity:
    """Token-sparsity skip (sparsity_thresh) in prefill attention."""

    def test_sparsity_disabled_by_default(self):
        """Default GQAttention has sparsity_thresh=0 (disabled)."""
        attn  = GQAttention(64, 4, 2, 16)
        assert attn.sparsity_thresh == 0.0

    def test_sparsity_no_nan(self):
        """Enabling token sparsity does not produce NaN."""
        attn  = GQAttention(64, 4, 2, 16, sparsity_thresh=0.01)
        cache = KVCache(n_layers=1, n_kv_heads=2, head_dim=16,
                        window=64, sink=4)
        # Prefill (seq > 1) triggers sparsity path
        x = np.random.randn(1, 8, 64).astype(np.float32)
        out = attn.forward(x, cache, layer_idx=0)
        assert out.shape == (1, 8, 64)
        assert np.all(np.isfinite(out))

    def test_sparsity_decode_unchanged(self):
        """Sparsity thresh is only applied during prefill (seq > 1);
        single-token decode runs the normal path."""
        attn  = GQAttention(64, 4, 2, 16, sparsity_thresh=0.99)
        cache = KVCache(n_layers=1, n_kv_heads=2, head_dim=16,
                        window=64, sink=4)
        # Warm up with prefill
        x_pre = np.random.randn(1, 4, 64).astype(np.float32)
        attn.forward(x_pre, cache, layer_idx=0)
        # Decode step — must not crash with extreme thresh
        x_dec = np.random.randn(1, 1, 64).astype(np.float32)
        out   = attn.forward(x_dec, cache, layer_idx=0)
        assert out.shape == (1, 1, 64)
        assert np.all(np.isfinite(out))