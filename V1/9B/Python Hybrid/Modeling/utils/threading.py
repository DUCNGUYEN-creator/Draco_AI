# DracoAI V1 — modeling/utils/threading.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""Thread-safe helpers used across runtime/ and kv_cache/."""
from __future__ import annotations
import threading
from contextlib import contextmanager
from typing import Generator

__all__ = ["RWLock", "atomic_counter", "once"]


class RWLock:
    """
    Simple readers-writer lock.
    Multiple readers can hold simultaneously; writers are exclusive.

    Usage::

        lock = RWLock()
        with lock.read():  ...
        with lock.write(): ...
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers    = 0

    @contextmanager
    def read(self) -> Generator:
        with self._read_ready:
            self._readers += 1
        try:
            yield
        finally:
            with self._read_ready:
                self._readers -= 1
                if self._readers == 0:
                    self._read_ready.notify_all()

    @contextmanager
    def write(self) -> Generator:
        with self._read_ready:
            while self._readers > 0:
                self._read_ready.wait()
            yield


class atomic_counter:
    """Thread-safe integer counter."""

    def __init__(self, initial: int = 0):
        self._val  = initial
        self._lock = threading.Lock()

    def increment(self, n: int = 1) -> int:
        with self._lock:
            self._val += n
            return self._val

    def decrement(self, n: int = 1) -> int:
        with self._lock:
            self._val -= n
            return self._val

    def get(self) -> int:
        with self._lock:
            return self._val

    def reset(self, val: int = 0):
        with self._lock:
            self._val = val

    def __repr__(self) -> str:
        return f"atomic_counter({self.get()})"


def once(fn):
    """
    Decorator: run fn at most once (thread-safe).
    Subsequent calls return the cached result.
    """
    _lock   = threading.Lock()
    _done   = False
    _result = None

    def wrapper(*args, **kwargs):
        nonlocal _done, _result
        if _done:
            return _result
        with _lock:
            if not _done:
                _result = fn(*args, **kwargs)
                _done   = True
        return _result
    return wrapper