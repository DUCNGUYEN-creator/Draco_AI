# DracoAI V1 — modeling/quant/int4.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
from __future__ import annotations
import numpy as np
from typing import Optional

__all__ = ["QuantizedLinear"]


class QuantizedLinear:
    """
    Weight-only quantized linear layer (INT8 or INT4 packed).

    forward(x) always returns float32 output.  Dequantization is
    cached to avoid repeated unpacking on every inference step.

    _transposed flag: when True, the stored weight matrix was transposed
    before quantization (so forward() uses @ W.T implicitly via dequantize).
    """

    def __init__(self, cache_dequant: bool = True):
        self.W_q:        Optional[np.ndarray] = None
        self.scale:      Optional[np.ndarray] = None
        self.zero:       Optional[np.ndarray] = None
        self.quant:      str = "int8"
        self.in_feat:    int = 0
        self.out_feat:   int = 0
        self.group_size: int = 128
        self._cache_dequant: bool = cache_dequant
        self._cached_W:      Optional[np.ndarray] = None
        self._transposed:    bool = False

    @classmethod
    def from_float(cls, W: np.ndarray, quant: str = "int8",
                   group_size: int = 128, cache_dequant: bool = True) -> "QuantizedLinear":
        ql = cls(cache_dequant=cache_dequant)
        ql.out_feat, ql.in_feat = W.shape
        ql.quant      = quant
        ql.group_size = group_size
        W32 = W.astype(np.float32)

        if quant == "int8":
            scale = np.abs(W32).max(axis=1, keepdims=True) / 127.0 + 1e-9
            ql.W_q   = np.clip(np.round(W32 / scale), -127, 127).astype(np.int8)
            ql.scale = scale.squeeze(1).astype(np.float32)
            ql.zero  = None

        elif quant == "int4":
            n_groups = ql.in_feat // group_size
            if n_groups == 0:
                raise ValueError(f"group_size={group_size} > in_feat={ql.in_feat}")
            usable = n_groups * group_size
            W_use  = W32[:, :usable]
            out    = W_use.shape[0]
            W_g    = W_use.reshape(out, n_groups, group_size)

            w_min   = W_g.min(axis=-1)
            w_max   = W_g.max(axis=-1)
            scale_g = (w_max - w_min) / 15.0 + 1e-9
            zero_g  = w_min

            W_int = np.clip(
                np.round((W_g - zero_g[:, :, None]) / scale_g[:, :, None]), 0, 15
            ).astype(np.uint8)

            W_flat = W_int.reshape(out, usable)
            if usable % 2 != 0:
                W_flat = np.pad(W_flat, ((0, 0), (0, 1)))
            W_packed = (W_flat[:, 0::2] & 0x0F) | ((W_flat[:, 1::2] & 0x0F) << 4)
            ql.W_q   = W_packed.astype(np.uint8)
            ql.scale = scale_g.astype(np.float32)
            ql.zero  = zero_g.astype(np.float32)
        else:
            raise ValueError(f"Unknown quant={quant!r}")
        return ql

    def dequantize(self) -> np.ndarray:
        if self._cached_W is not None:
            return self._cached_W
        W = self._dequantize_raw()
        if self._cache_dequant:
            self._cached_W = W
        return W

    def _dequantize_raw(self) -> np.ndarray:
        if self.quant == "int8":
            return self.W_q.astype(np.float32) * self.scale[:, None]
        elif self.quant == "int4":
            n_groups = self.scale.shape[1]
            out      = self.out_feat
            usable   = n_groups * self.group_size
            lo = (self.W_q & 0x0F).astype(np.float32)
            hi = ((self.W_q >> 4) & 0x0F).astype(np.float32)
            n_packed = self.W_q.shape[1]
            W_r = np.empty((out, n_packed * 2), dtype=np.float32)
            W_r[:, 0::2] = lo
            W_r[:, 1::2] = hi
            W_r = W_r[:, :usable].reshape(out, n_groups, self.group_size)
            return (W_r * self.scale[:, :, None] + self.zero[:, :, None]).reshape(out, usable)
        raise ValueError(f"Unknown quant: {self.quant!r}")

    def invalidate_cache(self):
        self._cached_W = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x @ W.T — works for both transposed and non-transposed storage."""
        W = self.dequantize()
        return x.astype(np.float32) @ W.T

    def save(self, path: str):
        d = dict(W_q=self.W_q, scale=self.scale,
                 quant=np.array(self.quant), in_feat=np.array(self.in_feat),
                 out_feat=np.array(self.out_feat), group_size=np.array(self.group_size),
                 transposed=np.array(int(self._transposed)),
                 cache_dequant=np.array(int(self._cache_dequant)))
        if self.zero is not None:
            d["zero"] = self.zero
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str, cache_dequant: bool = True) -> "QuantizedLinear":
        fname = path if path.endswith(".npz") else path + ".npz"
        data  = np.load(fname)
        if "cache_dequant" in data:
            cache_dequant = bool(int(data["cache_dequant"]))
        ql = cls(cache_dequant=cache_dequant)
        ql.W_q        = data["W_q"]
        ql.scale      = data["scale"]
        ql.quant      = str(data["quant"])
        ql.in_feat    = int(data["in_feat"])
        ql.out_feat   = int(data["out_feat"])
        ql.group_size = int(data["group_size"])
        ql.zero       = data["zero"] if "zero" in data else None
        ql._transposed = bool(int(data["transposed"])) if "transposed" in data else False
        ql._cached_W  = None
        return ql

    def __repr__(self) -> str:
        return (f"QuantizedLinear(quant={self.quant!r}, "
                f"shape=({self.out_feat},{self.in_feat}), gs={self.group_size})")