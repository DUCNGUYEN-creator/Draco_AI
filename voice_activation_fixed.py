#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
VOICE ACTIVATION - FIXED FINAL (FULL VERSION)
- Đã sửa lỗi: 'unexpected keyword argument pause_threshold'
- Giữ nguyên 100%: Lazy Loading, Threading, Queue, Auto-Unload.
- Không cắt bớt chức năng nào.
"""
from pathlib import Path
import threading
import time
import queue
from typing import Optional, Callable, Dict, Any
import sys

# Thêm đường dẫn để import được lazy_loader nằm cùng thư mục
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lazy_loader import get_lazy_loader
except ImportError:
    # Fallback an toàn nếu chạy test riêng lẻ
    def get_lazy_loader():
        return None


class DracoVoiceActivation:
    """Voice activation system với lazy loading"""

    def __init__(self, config):
        self.config = config
        self.keyword = config.get("voice.wake_word", "hey draco").lower()

        # State management
        self.listening = False
        self.stop_event = threading.Event()

        # Audio buffer queue
        self.audio_queue = queue.Queue(maxsize=5)

        # Thread containers
        self.listener_thread = None
        self.processor_thread = None

        # Lazy loader manager
        self.lazy_loader = get_lazy_loader()

        # Callbacks events
        self.on_activation = None
        self.on_error = None

        # Session tracking
        self.session_id = f"voice_{int(time.time())}"

        # Đăng ký ngay khi khởi tạo
        self._register_components()

    def _register_components(self):
        """Đăng ký speech recognition components vào Lazy Loader"""
        if not self.lazy_loader:
            print("⚠️ Warning: Lazy Loader not found!")
            return

        def load_speech_recognizer():
            """Chỉ load thư viện SpeechRecognition khi cần"""
            try:
                import speech_recognition as sr
                recognizer = sr.Recognizer()
                print("✅ [Voice] Speech Recognizer loaded into RAM")
                return recognizer
            except ImportError:
                print("❌ [Voice] speech_recognition library missing")
                raise

        def load_microphone():
            """Chỉ load driver Microphone khi cần"""
            try:
                import speech_recognition as sr
                # Chọn microphone mặc định của hệ thống
                microphone = sr.Microphone()
                print("✅ [Voice] Microphone driver loaded")
                return microphone
            except Exception as e:
                print(f"❌ [Voice] Microphone error: {e}")
                raise

        # Đăng ký với ước lượng RAM (MB)
        self.lazy_loader.register_component(
            name="speech_recognizer",
            loader_func=load_speech_recognizer,
            estimated_memory_mb=15
        )

        self.lazy_loader.register_component(
            name="microphone",
            loader_func=load_microphone,
            estimated_memory_mb=5
        )

    def start(self):
        """Bắt đầu quá trình nghe (Start Threads)"""
        if self.listening:
            return

        print("🎤 [Voice] Starting Voice Activation System...")

        self.listening = True
        self.stop_event.clear()

        # Khởi tạo 2 luồng riêng biệt: 1 nghe, 1 xử lý
        self.listener_thread = threading.Thread(
            target=self._listener_loop,
            name="VoiceListener",
            daemon=True
        )

        self.processor_thread = threading.Thread(
            target=self._processor_loop,
            name="VoiceProcessor",
            daemon=True
        )

        self.listener_thread.start()
        # Đợi một chút để luồng nghe ổn định
        time.sleep(0.5)
        self.processor_thread.start()

        print("✅ [Voice] Activation threads started")
        return True

    def stop(self):
        """Dừng an toàn và giải phóng RAM"""
        self.listening = False
        self.stop_event.set()

        # Dọn sạch hàng đợi
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except:
                pass

        # Chờ các luồng kết thúc (timeout 2s để tránh treo)
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        if self.processor_thread:
            self.processor_thread.join(timeout=2)

        # Quan trọng: Giải phóng RAM ngay lập tức
        if self.lazy_loader:
            self.lazy_loader.unload_all()

        print("🎤 [Voice] System stopped & RAM cleared")

    def _listener_loop(self):
        """Luồng lắng nghe (Background Listening)"""
        try:
            # Lấy components từ Lazy Loader (Lúc này mới nạp vào RAM)
            microphone = self.lazy_loader.get_component("microphone")
            recognizer = self.lazy_loader.get_component("speech_recognizer")

            print("[Voice] Calibrating microphone for ambient noise...")
            with microphone as source:
                # Lọc tiếng ồn môi trường trong 1 giây
                recognizer.adjust_for_ambient_noise(source, duration=1)

            print(f"[Voice] Listening for keyword: '{self.keyword}'")

            # ============================================================
            # KHU VỰC SỬA LỖI (FIXED ZONE)
            # ============================================================
            # 1. Cài đặt các thông số nhạy trực tiếp vào object
            recognizer.pause_threshold = 0.8  # Thời gian nghỉ để ngắt câu
            recognizer.energy_threshold = 300  # Độ nhạy âm thanh
            recognizer.dynamic_energy_threshold = True

            # 2. Gọi hàm listen_in_background (ĐÃ BỎ tham số gây lỗi)
            stop_listening = recognizer.listen_in_background(
                microphone,
                self._audio_callback,
                phrase_time_limit=5  # Giới hạn mỗi câu nói 5s
            )
            # ============================================================

            # Giữ thread sống cho đến khi có lệnh dừng
            while not self.stop_event.is_set():
                time.sleep(0.5)

            # Dừng nghe khi thoát
            stop_listening(wait_for_stop=False)

        except Exception as e:
            print(f"❌ [Voice Listener Error]: {e}")
            if self.on_error:
                self.on_error(str(e))

    def _audio_callback(self, recognizer, audio):
        """Callback khi mic bắt được âm thanh"""
        try:
            # Chỉ đẩy vào hàng đợi nếu chưa đầy (tránh tràn RAM)
            if self.audio_queue.qsize() < 5:
                self.audio_queue.put(audio)
        except Exception as e:
            print(f"Audio queue error: {e}")

    def _processor_loop(self):
        """Luồng xử lý âm thanh sang văn bản (STT)"""
        while not self.stop_event.is_set():
            try:
                # Lấy audio từ hàng đợi (timeout 2s để check stop_event)
                audio = self.audio_queue.get(timeout=2.0)

                # Cần recognizer để giải mã
                recognizer = self.lazy_loader.get_component("speech_recognizer")

                try:
                    # Sử dụng Google Speech Recognition (Online nhưng nhẹ)
                    # Hoặc có thể thay bằng Whisper Local ở đây sau này
                    text = recognizer.recognize_google(audio).lower()

                    # Debug log (có thể comment lại nếu muốn gọn)
                    # print(f"🎤 [Heard]: {text}")

                    # Kiểm tra từ khóa đánh thức
                    if self.keyword in text:
                        self._handle_activation(text)

                except Exception:
                    # Không nhận diện được hoặc lỗi mạng -> Bỏ qua
                    pass

                self.audio_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processor loop error: {e}")

    def _handle_activation(self, text):
        """Xử lý khi phát hiện từ khóa"""
        print(f"🔊 [ACTIVATION TRIGGERED]: {text}")

        if self.on_activation:
            self.on_activation(text)

        # Memory Guard: Tự động unload Mic và Recognizer sau 30s không dùng
        if self.lazy_loader:
            print("[Memory] Scheduling voice components unload in 30s...")
            self.lazy_loader.schedule_unload("speech_recognizer", timeout=30)
            self.lazy_loader.schedule_unload("microphone", timeout=30)

    def set_callbacks(self, on_activation=None, on_error=None):
        """Thiết lập hàm gọi lại từ main"""
        self.on_activation = on_activation
        self.on_error = on_error

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái hệ thống (Cho GUI hiển thị)"""
        loader_status = {}
        if self.lazy_loader:
            loader_status = self.lazy_loader.get_status()

        return {
            "listening": self.listening,
            "keyword": self.keyword,
            "session_id": self.session_id,
            "loader_status": loader_status
        }

    def cleanup(self):
        """Hàm dọn dẹp khi tắt ứng dụng"""
        self.stop()
        print("Voice Activation resources cleaned up")


# Global Singleton Instance
_voice_instance = None


def get_voice_activation(config=None):
    """Hàm lấy instance toàn cục (Singleton)"""
    global _voice_instance
    if _voice_instance is None and config:
        _voice_instance = DracoVoiceActivation(config)
    return _voice_instance