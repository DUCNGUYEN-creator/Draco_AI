# DracoAI V1 — modeling/testing/test_quant.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Unit tests for INT8 / INT4 quantization.

FIXES (this revision):
  ✅ FIX-UNUSED-IMPORT-DEQUANTIZE-INT4 : removed unused `dequantize_int4`
     import.  The function is tested indirectly via QuantizedLinear (INT4
     path), but no test in this file calls dequantize_int4 directly.
     dequantize_int8 IS used in test_dequant_roundtrip_int8 and is kept.
"""
import numpy as np
import pytest
from ..quant.int4   import QuantizedLinear
from ..quant.scales import (compute_int8_scale, compute_int4_scale_zero,
                            dequantize_int8)


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
        np.testing.assert_allclose(W_deq, W[:, :W_deq.shape[1]], atol=0.1)

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