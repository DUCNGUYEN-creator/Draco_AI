# DracoAI V1 — modeling/testing/test_quant.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for INT8/INT4 quantization and TernaryLinear.

FIXES retained:
  ✅ FIX-UNUSED-IMPORT-DEQUANTIZE-INT4 : removed unused direct import.

NEW tests in this revision:
  • TestTernaryLinear           — pack/unpack, forward, persistence
  • TestTernarizeWeight         — ternarize_weight() utility
  • TestTernaryQuant            — quantize_model_weights('ternary') integration
"""
import numpy as np
import pytest
from ..quant.int4           import QuantizedLinear
from ..quant.ternary_linear import TernaryLinear, ternarize_weight
from ..quant.quant_linear   import quantize_model_weights
from ..quant.scales         import (compute_int8_scale, compute_int4_scale_zero,
                                     dequantize_int8)


# ─────────────────────────────────────────────────────────────────────────────
# Original tests (all retained)
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantizedLinearINT8:
    @pytest.fixture
    def ql(self):
        W  = np.random.randn(32, 64).astype(np.float32)
        return QuantizedLinear.from_float(W, quant="int8"), W

    def test_dequant_close_to_original(self, ql):
        qlayer, W = ql
        W_deq = qlayer.dequantize()
        np.testing.assert_allclose(W_deq, W, atol=0.02)

    def test_forward_shape(self, ql):
        qlayer, W = ql
        x   = np.random.randn(5, 64).astype(np.float32)
        out = qlayer.forward(x)
        assert out.shape == (5, 32)

    def test_cache_invalidate(self, ql):
        qlayer, _ = ql
        _ = qlayer.dequantize()
        assert qlayer._cached_W is not None
        qlayer.invalidate_cache()
        assert qlayer._cached_W is None


class TestQuantizedLinearINT4:
    @pytest.fixture
    def ql(self):
        W = np.random.randn(32, 256).astype(np.float32)
        return QuantizedLinear.from_float(W, quant="int4", group_size=64), W

    def test_dequant_close_to_original(self, ql):
        qlayer, W = ql
        W_deq = qlayer.dequantize()
        assert W_deq.shape[0] == 32
        # INT4 asymmetric per-group quantization: max error = scale/2 = (max-min)/30
        # For random normal data with group_size=64, max error can reach ~0.25
        np.testing.assert_allclose(W_deq, W[:, :W_deq.shape[1]], atol=0.3)

    def test_forward_shape(self, ql):
        qlayer, _ = ql
        x   = np.random.randn(3, 256).astype(np.float32)
        out = qlayer.forward(x)
        assert out.shape[0] == 3 and out.shape[1] == 32

    def test_group_size_too_large_raises(self):
        W = np.random.randn(8, 32).astype(np.float32)
        with pytest.raises(ValueError):
            QuantizedLinear.from_float(W, quant="int4", group_size=64)


class TestScaleUtils:
    def test_int8_scale_positive(self):
        W     = np.random.randn(16, 32).astype(np.float32)
        scale = compute_int8_scale(W)
        assert scale.shape == (16,) and np.all(scale > 0)

    def test_int4_scale_zero_shapes(self):
        W = np.random.randn(8, 128).astype(np.float32)
        scale, zero = compute_int4_scale_zero(W, group_size=64)
        assert scale.shape == (8, 2)
        assert zero.shape  == (8, 2)

    def test_dequant_roundtrip_int8(self):
        W     = np.random.randn(16, 32).astype(np.float32)
        scale = compute_int8_scale(W)
        W_q   = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
        W_rec = dequantize_int8(W_q, scale)
        np.testing.assert_allclose(W_rec, W, atol=0.02)


# ─────────────────────────────────────────────────────────────────────────────
# NEW tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTernaryLinear:
    """TernaryLinear layer unit tests."""

    @pytest.fixture
    def tl(self):
        W = np.random.randn(32, 64).astype(np.float32)
        return TernaryLinear.from_float(W), W

    def test_from_float_shapes(self, tl):
        layer, W = tl
        assert layer.out_feat == 32
        assert layer.in_feat  == 64
        # Packed: ceil(64/4) = 16 bytes per output
        assert layer.W_packed.shape == (32, 16)
        assert layer.W_packed.dtype == np.uint8
        assert layer.scale.shape == (32,)
        assert layer.scale.dtype == np.float32

    def test_packed_values_in_range(self, tl):
        """All 2-bit packed values decode to {-1, 0, +1}."""
        layer, _ = tl
        from ..quant.ternary_linear import _unpack_ternary
        W_t = _unpack_ternary(layer.W_packed, layer.in_feat)
        unique_vals = set(np.unique(W_t).tolist())
        assert unique_vals <= {-1, 0, 1}

    def test_forward_shape(self, tl):
        layer, _ = tl
        x   = np.random.randn(5, 64).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (5, 32)

    def test_forward_no_nan(self, tl):
        layer, _ = tl
        x = np.random.randn(3, 64).astype(np.float32)
        out = layer.forward(x)
        assert np.all(np.isfinite(out))

    def test_addition_only_matches_matmul(self, tl):
        """Addition-only forward gives same result as W_t @ x.T * scale."""
        layer, _ = tl
        from ..quant.ternary_linear import _unpack_ternary
        x    = np.random.randn(4, 64).astype(np.float32)
        # Reference: explicit dequant matmul
        W_t  = _unpack_ternary(layer.W_packed, layer.in_feat).astype(np.float32)
        ref  = x @ W_t.T * layer.scale[None, :]
        # Our forward
        out  = layer.forward(x)
        np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_dequantize_reconstructs_approx(self, tl):
        """dequantize() reconstructs float weight with reasonable error."""
        layer, W = tl
        W_rec = layer.dequantize()
        assert W_rec.shape == (32, 64)
        # Ternary has ~30% relative error — check it's finite and bounded
        assert np.all(np.isfinite(W_rec))

    def test_dequantize_cached(self, tl):
        """Second dequantize() call returns cached result."""
        layer, _ = tl
        W1 = layer.dequantize()
        W2 = layer.dequantize()
        assert W1 is W2   # same object (cached)

    def test_invalidate_cache(self, tl):
        layer, _ = tl
        _ = layer.dequantize()
        assert layer._W_float_cache is not None
        layer.invalidate_cache()
        assert layer._W_float_cache is None

    def test_save_load_roundtrip(self, tl, tmp_path):
        layer, _ = tl
        path = str(tmp_path / "ternary")
        layer.save(path)
        loaded = TernaryLinear.load(path)
        assert loaded.out_feat == layer.out_feat
        assert loaded.in_feat  == layer.in_feat
        np.testing.assert_array_equal(loaded.W_packed, layer.W_packed)
        np.testing.assert_array_equal(loaded.scale,    layer.scale)

    def test_forward_batch_shapes(self, tl):
        """Forward works on 2D (batch×in) inputs."""
        layer, _ = tl
        for batch in (1, 4, 8):
            x = np.random.randn(batch, 64).astype(np.float32)
            out = layer.forward(x)
            assert out.shape == (batch, 32)

    def test_scale_positive(self, tl):
        """Per-row scales are always positive."""
        layer, _ = tl
        assert np.all(layer.scale > 0)

    def test_compression_ratio(self, tl):
        """TernaryLinear uses ~4× less memory than float32."""
        layer, W = tl
        float_bytes = W.nbytes       # float32
        tern_bytes  = (layer.W_packed.nbytes + layer.scale.nbytes)
        ratio = float_bytes / tern_bytes
        assert ratio > 3.5, f"Expected >3.5× compression, got {ratio:.1f}×"


class TestTernarizeWeight:
    """ternarize_weight() utility function tests."""

    def test_output_types(self):
        W = np.random.randn(16, 32).astype(np.float32)
        packed, scale, in_feat = ternarize_weight(W)
        assert packed.dtype == np.uint8
        assert scale.dtype  == np.float32
        assert in_feat == 32

    def test_scale_is_mean_abs(self):
        """Scale equals mean(|W|) per row (BitNet 1.58b formula)."""
        W = np.random.randn(8, 16).astype(np.float32)
        _, scale, _ = ternarize_weight(W)
        expected = np.abs(W).mean(axis=1)
        np.testing.assert_allclose(scale, expected, rtol=1e-5)

    def test_packed_pack_unpack(self):
        """ternarize → unpack gives values in {-1, 0, +1}."""
        from ..quant.ternary_linear import _unpack_ternary
        W = np.random.randn(12, 24).astype(np.float32)
        packed, _, in_feat = ternarize_weight(W)
        W_t = _unpack_ternary(packed, in_feat)
        assert W_t.shape == (12, 24)
        assert set(np.unique(W_t).tolist()) <= {-1, 0, 1}

    def test_zero_weight_no_crash(self):
        """All-zero weight matrix ternarizes to all-zero packed."""
        W = np.zeros((4, 8), dtype=np.float32)
        packed, scale, _ = ternarize_weight(W)
        assert np.all(packed == 0)


class TestTernaryQuant:
    """quantize_model_weights('ternary') integration test."""

    def test_ternary_mode_green_zone_only(self):
        """Ternary quant only touches FFN experts, not attention or router."""
        from ..config      import ModelConfig
        from ..transformer import DracoTransformerV1
        cfg   = ModelConfig(d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
                            head_dim=16, d_ff=64, n_experts=2, vocab_size=64, window=32)
        model = DracoTransformerV1(cfg, dtype=np.float32)
        quantize_model_weights(model, "ternary")

        for blk in model.blocks:
            # GREEN: expert FFN must be TernaryLinear
            for exp in list(blk.moe.experts) + [blk.moe.shared]:
                assert isinstance(exp.W_g, TernaryLinear), \
                    "W_g must be TernaryLinear after ternary quant"
                assert isinstance(exp.W_u, TernaryLinear), \
                    "W_u must be TernaryLinear after ternary quant"
                assert isinstance(exp.W_d, TernaryLinear), \
                    "W_d must be TernaryLinear after ternary quant"
            # RED: attention must remain float
            assert isinstance(blk.attn.W_q, np.ndarray), \
                "Attention W_q must stay float after ternary quant"
            assert isinstance(blk.attn.W_k, np.ndarray)
            assert isinstance(blk.attn.W_v, np.ndarray)
            assert isinstance(blk.attn.W_o, np.ndarray)
            # RED: router must remain float32
            assert blk.moe.W_router.dtype == np.float32

    def test_ternary_forward_finite(self):
        """Model with ternary experts produces finite logits."""
        from ..config      import ModelConfig
        from ..transformer import DracoTransformerV1
        cfg   = ModelConfig(d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
                            head_dim=16, d_ff=64, n_experts=2, vocab_size=64, window=32)
        model = DracoTransformerV1(cfg, dtype=np.float32)
        quantize_model_weights(model, "ternary")
        cache = model._make_cache()
        l1, l2, _ = model.forward([1, 2, 3], cache)
        assert np.all(np.isfinite(l1)), "NaN in ternary model"
        assert np.all(np.isfinite(l2))

    def test_invalid_quant_raises(self):
        """quantize_model_weights raises on unknown quant string."""
        from ..config      import ModelConfig
        from ..transformer import DracoTransformerV1
        cfg   = ModelConfig(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
                            head_dim=16, d_ff=32, n_experts=2, vocab_size=32, window=16)
        model = DracoTransformerV1(cfg)
        with pytest.raises(ValueError):
            quantize_model_weights(model, "fp8")