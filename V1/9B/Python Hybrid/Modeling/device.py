# DracoAI V1 — modeling/device.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
Hardware detection and backend selection.

This is the SINGLE source of truth for "what can this machine do".
Everything else (dtypes.py, kernels/, layers/block.py) queries device.py
rather than sniffing hardware themselves.

Architecture rule:
    device.py → constants.py only.
    Nothing imports device.py at module level from inside ops/ or layers/
    (to avoid circular imports). Use lazy get_capability() calls at runtime.

FIXES (this revision):
  ✅ FIX-AVX2-MACOS: /proc/cpuinfo does not exist on macOS.  AVX2 is now
     detected via `sysctl -n machdep.cpu.features` on darwin and via
     /proc/cpuinfo on Linux.  This enables correct float16 selection on
     Apple silicon and x86 Macs.
"""
from __future__ import annotations
import logging
import sys
from dataclasses import dataclass, field
from typing import List

from .constants import KERNEL_NUMPY, KERNEL_TRITON, KERNEL_NUMBA

__all__ = [
    "HardwareCapability", "detect_hardware_capability",
    "get_capability", "get_optimal_backend",
    "has_triton", "has_numba", "has_cuda",
]

logger = logging.getLogger(__name__)

# ── Module-level singleton (lazy, populated on first call) ─────────────
_CAPABILITY: "HardwareCapability | None" = None


@dataclass
class HardwareCapability:
    """
    Snapshot of hardware capabilities detected at startup.

    Fields
    ------
    has_cuda    : CUDA GPU available (via CuPy or PyTorch probe)
    has_triton  : triton package importable AND CUDA available
    has_numba   : numba package importable
    has_avx2    : x86 AVX2 SIMD (used for NumPy vectorisation hints)
    cuda_devices: list of (device_idx, name, vram_gb) tuples
    optimal_backend: KERNEL_NUMPY | KERNEL_TRITON | KERNEL_NUMBA
    """
    has_cuda:         bool = False
    has_triton:       bool = False
    has_numba:        bool = False
    has_avx2:         bool = False
    cuda_devices:     List[tuple] = field(default_factory=list)
    optimal_backend:  str  = KERNEL_NUMPY


def _probe_avx2_linux() -> bool:
    """Detect AVX2 on Linux via /proc/cpuinfo."""
    try:
        import subprocess
        result = subprocess.run(
            ["grep", "-m1", "avx2", "/proc/cpuinfo"],
            capture_output=True, timeout=1)
        return result.returncode == 0 and b"avx2" in result.stdout
    except Exception:
        return False


def _probe_avx2_macos() -> bool:
    """
    Detect AVX2 on macOS via sysctl.
    Works for both x86_64 Macs and Apple Silicon (which always has NEON,
    but reports AVX2 as absent — that is correct and safe).
    """
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.leaf7_features"],
            capture_output=True, timeout=1)
        if result.returncode == 0 and b"AVX2" in result.stdout.upper():
            return True
        # Older macOS exposes features via machdep.cpu.features
        result2 = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.features"],
            capture_output=True, timeout=1)
        return result2.returncode == 0 and b"AVX2" in result2.stdout.upper()
    except Exception:
        return False


def detect_hardware_capability() -> HardwareCapability:
    """
    Probe hardware and return a HardwareCapability snapshot.
    Safe to call multiple times — result is deterministic.
    """
    cap = HardwareCapability()

    # ── CUDA probe ────────────────────────────────────────────────────
    try:
        import importlib
        cupy = importlib.import_module("cupy")
        n = cupy.cuda.runtime.getDeviceCount()
        if n > 0:
            cap.has_cuda = True
            for i in range(n):
                props = cupy.cuda.runtime.getDeviceProperties(i)
                name  = props["name"].decode() if isinstance(props["name"], bytes) \
                        else str(props["name"])
                vram  = props.get("totalGlobalMem", 0) / 1024 ** 3
                cap.cuda_devices.append((i, name, round(vram, 2)))
    except Exception:
        try:
            import importlib
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                cap.has_cuda = True
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    vram = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                    cap.cuda_devices.append((i, name, round(vram, 2)))
        except Exception:
            pass

    # ── Triton probe ──────────────────────────────────────────────────
    if cap.has_cuda:
        try:
            import importlib
            importlib.import_module("triton")
            cap.has_triton = True
        except Exception:
            pass

    # ── Numba probe ───────────────────────────────────────────────────
    try:
        import importlib
        importlib.import_module("numba")
        cap.has_numba = True
    except Exception:
        pass

    # ── AVX2 probe — platform-aware ───────────────────────────────────
    # ✅ FIX-AVX2-MACOS: /proc/cpuinfo is Linux-only; use sysctl on macOS.
    if sys.platform in ("linux", "linux2"):
        cap.has_avx2 = _probe_avx2_linux()
    elif sys.platform == "darwin":
        cap.has_avx2 = _probe_avx2_macos()
    # Windows: no simple cross-version method; leave has_avx2=False (safe)

    # ── Choose optimal backend ────────────────────────────────────────
    if cap.has_triton:
        cap.optimal_backend = KERNEL_TRITON
    elif cap.has_numba:
        cap.optimal_backend = KERNEL_NUMBA
    else:
        cap.optimal_backend = KERNEL_NUMPY

    logger.info(
        "[DracoAI] Hardware: cuda=%s triton=%s numba=%s avx2=%s → backend=%s",
        cap.has_cuda, cap.has_triton, cap.has_numba,
        cap.has_avx2, cap.optimal_backend)
    return cap


def get_capability() -> HardwareCapability:
    """Return the module-level capability singleton (lazy init on first call)."""
    global _CAPABILITY
    if _CAPABILITY is None:
        _CAPABILITY = detect_hardware_capability()
    return _CAPABILITY


def get_optimal_backend() -> str:
    """Shortcut: return KERNEL_* string for the best available backend."""
    return get_capability().optimal_backend


def has_triton() -> bool:
    return get_capability().has_triton


def has_numba() -> bool:
    return get_capability().has_numba


def has_cuda() -> bool:
    return get_capability().has_cuda