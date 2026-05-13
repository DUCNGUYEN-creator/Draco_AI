# DracoAI V1 — modeling/utils/memory.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Memory tracking utilities. Platform-safe RSS / peak-RSS queries."""
from __future__ import annotations
import sys
from typing import Optional

__all__ = ["get_rss_mb", "format_bytes", "estimate_array_mb"]


def get_rss_mb() -> Optional[float]:
    """Return current process RSS in MB, or None if unavailable."""
    try:
        if sys.platform == "win32":
            import ctypes
            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb",                          ctypes.c_ulong),
                    ("PageFaultCount",              ctypes.c_ulong),
                    ("PeakWorkingSetSize",           ctypes.c_size_t),
                    ("WorkingSetSize",               ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage",      ctypes.c_size_t),
                    ("QuotaPagedPoolUsage",          ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage",   ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage",       ctypes.c_size_t),
                    ("PagefileUsage",                ctypes.c_size_t),
                    ("PeakPagefileUsage",            ctypes.c_size_t),
                ]
            pmc = _PMC()
            pmc.cb = ctypes.sizeof(pmc)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc), ctypes.sizeof(pmc))
            return pmc.WorkingSetSize / (1024 * 1024)
        else:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)
    except Exception:
        return None


def format_bytes(n: int) -> str:
    """Human-readable byte count: 1536 → '1.5 KB'"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def estimate_array_mb(shape: tuple, dtype_itemsize: int = 4) -> float:
    """Estimate memory for a NumPy array in MB."""
    import math
    n = math.prod(shape)
    return (n * dtype_itemsize) / (1024 * 1024)