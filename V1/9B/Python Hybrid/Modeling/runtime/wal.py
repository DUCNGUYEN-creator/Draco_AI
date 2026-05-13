# DracoAI V1 — modeling/runtime/wal.py
# Copyright (C) 2026  Draco Studio and DUCNGUYEN-creator — GPL v3
"""
WriteAheadLog — per-token fault-tolerant journal.

Uses context manager + __del__ for guaranteed resource cleanup.
Each record: [4-byte int32 token_id][4-byte float32 timestamp].

FIXES (this revision):
  ✅ FIX-WAL-TOCTOU : _closed check moved inside the lock in both append()
     and flush().  Previously the check lived outside the lock, leaving a
     race window where a concurrent close() call could close the file handle
     between the check and the write, raising an unhandled ValueError.
"""
from __future__ import annotations
import struct, threading, time
from typing import List

from ..constants import WAL_FLUSH_INTERVAL

__all__ = ["WriteAheadLog"]


class WriteAheadLog:
    """
    Append-only binary log.

    Usage (context manager recommended)::

        with WriteAheadLog("session.wal") as wal:
            model.generate([...], wal=wal)
        tokens = WriteAheadLog.recover("session.wal")
    """
    _RECORD_SIZE = 8  # 4 bytes token_id + 4 bytes timestamp

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._fh   = open(path, "ab")
        self._n_written = 0
        self._closed = False

    def append(self, token_id: int):
        # ✅ FIX-WAL-TOCTOU: _closed check moved inside the lock so there is no
        # race window between checking _closed and writing to the file handle.
        # Previously a concurrent close() call could close _fh between the
        # outside check and the write, raising an unhandled ValueError.
        record = struct.pack("<if", int(token_id), float(time.perf_counter()))
        with self._lock:
            if self._closed:
                return
            self._fh.write(record)
            self._n_written += 1
            if self._n_written % WAL_FLUSH_INTERVAL == 0:
                self._fh.flush()

    def flush(self):
        # ✅ FIX-WAL-TOCTOU: same as append() — check inside lock.
        with self._lock:
            if self._closed:
                return
            self._fh.flush()

    def close(self):
        with self._lock:
            if not self._closed:
                self._fh.flush()
                self._fh.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def recover(path: str) -> List[int]:
        tokens = []
        try:
            with open(path, "rb") as f:
                while True:
                    rec = f.read(WriteAheadLog._RECORD_SIZE)
                    if len(rec) < WriteAheadLog._RECORD_SIZE:
                        break
                    token_id, _ = struct.unpack("<if", rec)
                    tokens.append(int(token_id))
        except FileNotFoundError:
            pass
        return tokens

    def __repr__(self) -> str:
        return f"WriteAheadLog(path={self._path!r}, written={self._n_written})"