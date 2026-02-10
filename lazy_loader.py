#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO LAZY LOADER - Smart resource management with on-demand loading
"""
import threading
import time
import gc
from typing import Dict, Any, Optional, Callable
from enum import Enum


class LoadState(Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class DracoLazyLoader:
    """Smart lazy loader với auto-unload để tiết kiệm RAM"""

    def __init__(self):
        # Khai báo lock ngay đầu tiên để PyCharm nhận diện được self.lock ở mọi nơi
        self.lock = threading.RLock()
        self.components = {}
        self.unload_timers = {}
        self.default_timeout = 60  # 60 giây không dùng sẽ nhả RAM

    def register_component(self, name: str, loader_func: Callable,
                           unloader_func: Callable = None,
                           estimated_memory_mb: float = 100):
        """Đăng ký linh kiện (như AI Model hoặc OCR)"""
        with self.lock:
            self.components[name] = {
                "loader": loader_func,
                "unloader": unloader_func,
                "state": LoadState.NOT_LOADED,
                "instance": None,
                "memory_mb": estimated_memory_mb,
                "last_used": 0,
                "access_count": 0
            }

    def get_component(self, name: str, force_reload: bool = False) -> Any:
        """Lấy linh kiện, tự động nạp nếu chưa có"""
        with self.lock:
            if name not in self.components:
                raise KeyError(f"Linh kiện {name} chưa được đăng ký!")

            comp = self.components[name]

            # Hủy đếm ngược nhả RAM khi bắt đầu sử dụng
            if name in self.unload_timers:
                self.unload_timers[name].cancel()

            # Nếu đang trong quá trình nạp, hãy đợi (tối đa 30s)
            if comp["state"] == LoadState.LOADING:
                wait_start = time.time()
                while comp["state"] == LoadState.LOADING:
                    if time.time() - wait_start > 30:
                        raise TimeoutError(f"Nạp {name} quá lâu, kiểm tra lại phần cứng!")
                    time.sleep(0.1)

            # Thực hiện nạp nếu cần
            if force_reload or comp["state"] in [LoadState.NOT_LOADED, LoadState.ERROR]:
                comp["state"] = LoadState.LOADING
                try:
                    instance = comp["loader"]()
                    comp["instance"] = instance
                    comp["state"] = LoadState.LOADED
                    comp["last_used"] = time.time()
                    comp["access_count"] += 1
                    print(f"✅ [Draco] Đã nạp thành công: {name}")
                    return instance
                except Exception as e:
                    comp["state"] = LoadState.ERROR
                    print(f"❌ [Draco] Lỗi khi nạp {name}: {e}")
                    raise

            comp["last_used"] = time.time()
            comp["access_count"] += 1
            return comp["instance"]

    def schedule_unload(self, name: str, timeout: int = None):
        """Lên lịch tự động giải phóng RAM"""
        if timeout is None:
            timeout = self.default_timeout

        with self.lock:
            if name in self.unload_timers:
                self.unload_timers[name].cancel()

            timer = threading.Timer(timeout, self._unload_if_idle, args=[name, timeout])
            timer.daemon = True
            self.unload_timers[name] = timer
            timer.start()

    def _unload_if_idle(self, name: str, timeout: int):
        """Kiểm tra và nhả RAM nếu linh kiện đang rảnh"""
        with self.lock:
            if name not in self.components:
                return
            comp = self.components[name]
            idle_time = time.time() - comp["last_used"]
            if idle_time >= timeout and comp["state"] == LoadState.LOADED:
                print(f"🔄 [Draco] Đang giải phóng RAM rảnh: {name} ({idle_time:.1f}s)")
                self.unload_component(name)

    def unload_component(self, name: str):
        """Giải phóng RAM ngay lập tức"""
        with self.lock:
            comp = self.components.get(name)
            if comp and comp["state"] == LoadState.LOADED and comp["unloader"]:
                try:
                    comp["unloader"](comp["instance"])
                    comp["instance"] = None
                    comp["state"] = LoadState.NOT_LOADED
                    gc.collect()  # Dọn rác hệ thống
                    print(f"✅ [Draco] Đã giải phóng RAM cho: {name}")
                except Exception as e:
                    print(f"❌ [Draco] Lỗi khi giải phóng {name}: {e}")

    def unload_all(self):
        """Dọn dẹp tất cả trước khi tắt máy"""
        with self.lock:
            for name in list(self.components.keys()):
                self.unload_component(name)
            for timer in self.unload_timers.values():
                timer.cancel()
            self.unload_timers.clear()

    def get_status(self) -> Dict[str, Any]:
        """Xem trạng thái hệ thống"""
        status = {}
        with self.lock:
            for name, comp in self.components.items():
                status[name] = {
                    "state": comp["state"].value,
                    "idle_seconds": time.time() - comp["last_used"] if comp["last_used"] > 0 else 0,
                    "access_count": comp["access_count"]
                }
        return status

    def cleanup(self):
        self.unload_all()


# --- KHAI BÁO BIẾN TOÀN CỤC CHUẨN IDE ---
# Thêm Optional giúp PyCharm biết biến này có thể là None hoặc DracoLazyLoader
_lazy_loader: Optional[DracoLazyLoader] = None


def get_lazy_loader() -> DracoLazyLoader:
    """Hàm lấy loader duy nhất (Singleton Pattern)"""
    global _lazy_loader
    if _lazy_loader is None:
        _lazy_loader = DracoLazyLoader()
    return _lazy_loader