#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO V15 ULTRA CONFIGURATION - Complete with all settings
"""
import os
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import hashlib


class DracoConfig:
    """Complete configuration manager với storage auto-detection"""

    def __init__(self, config_file: str = None):
        self.version = "V15.3.0-Ultra-Final"
        self.build_date = "2024-01-26"
        self.safe_mode = "--safe" in sys.argv
        self.cli_mode = "--cli" in sys.argv

        # Load default config
        self.config = self._load_default_config()

        # Tìm storage tốt nhất
        self.storage_path = self._find_optimal_storage()

        # Tạo cấu trúc thư mục
        self._create_storage_structure()

        # Load user config nếu có
        if config_file and os.path.exists(config_file):
            self._load_user_config(config_file)

        # Model hashes để verify integrity
        self.config["ai"]["model_hashes"] = self._load_model_hashes()

        # Performance settings
        self.config["performance"]["lazy_loading"] = True
        self.config["performance"]["model_unload_timeout"] = 60  # 1 phút

    def _load_model_hashes(self) -> Dict[str, str]:
        """Load model hashes từ file"""
        try:
            hash_file = Path(__file__).parent / "model_hashes.json"
            if hash_file.exists():
                with open(hash_file, 'r') as f:
                    return json.load(f)
        except:
            pass

        # Default fallback
        return {
            "gemma-2-2b-it-Q4_K_M": "a1b2c3d4e5f67890abcdef1234567890",
            "moondream2": "f1e2d3c4b5a67890fedcba9876543210",
            "whisper-tiny": "1234567890abcdef1234567890abcdef",
            "yolov4-tiny": "7890abcdef1234567890abcdef123456"
        }

    def _find_optimal_storage(self) -> Path:
        """Tìm storage tốt nhất"""
        try:
            partitions = psutil.disk_partitions(all=False)
            best_path = None
            best_score = -1

            for partition in partitions:
                try:
                    # Kiểm tra quyền ghi
                    test_path = Path(partition.mountpoint) / ".draco_test"
                    test_path.parent.mkdir(parents=True, exist_ok=True)
                    test_path.write_text("test")
                    test_path.unlink()

                    # Tính điểm
                    usage = psutil.disk_usage(partition.mountpoint)
                    score = usage.free

                    if score > best_score:
                        best_score = score
                        best_path = Path(partition.mountpoint)

                except (PermissionError, OSError):
                    continue

            # Fallback
            if best_path is None:
                if platform.system() == "Windows":
                    best_path = Path(os.environ.get("USERPROFILE", "C:\\Users\\Public"))
                else:
                    best_path = Path.home()

            # Tạo thư mục Draco
            draco_root = best_path / "DracoAI"
            draco_root.mkdir(parents=True, exist_ok=True)

            return draco_root

        except Exception as e:
            print(f"Storage detection error: {e}")
            return Path.cwd()

    def _create_storage_structure(self):
        """Tạo cấu trúc thư mục"""
        dirs = [
            "models", "bin", "database", "logs",
            "cache", "backups", "screenshots",
            "voice_recordings", "downloads"
        ]

        for dir_name in dirs:
            dir_path = self.storage_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "version": self.version,
            "build_date": self.build_date,
            "system": {
                "mode": "production",
                "auto_update": True,
                "backup_interval": 3600,
                "max_log_size_mb": 100,
                "safe_mode": self.safe_mode,
                "cli_mode": self.cli_mode
            },
            "ai": {
                "core_model": "gemma-2-2b-it-Q4_K_M",
                "vision_model": "moondream2",
                "stt_model": "whisper-tiny",
                "tts_engine": "pyttsx3",
                "enable_vision": True,
                "enable_voice": True,
                "enable_automation": True,
                "enable_search": True,
                "context_size": 4096,
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.95
            },
            "voice": {
                "wake_word": "hey draco",
                "language": "en-US",
                "energy_threshold": 300,
                "pause_threshold": 0.8,
                "phrase_time_limit": 5,
                "use_offline_stt": True,
                "tts_voice": "default",
                "tts_rate": 150,
                "require_confirmation": True
            },
            "vision": {
                "screen_capture_interval": 1.0,
                "ocr_engine": "tesseract",
                "object_detection": True,
                "face_recognition": False,
                "save_captures": False,
                "tesseract_path": None,
                "max_image_size_mb": 10
            },
            "automation": {
                "mouse_speed": 1.0,
                "keyboard_delay": 0.1,
                "max_automation_time": 300,
                "enable_screen_control": True,
                "enable_file_operations": False,
                "enable_app_control": True,
                "dangerous_actions_require_confirmation": True,
                "allowed_apps": [],
                "blocked_actions": ["delete", "format", "shutdown"]
            },
            "search": {
                "engine": "duckduckgo",
                "max_results": 5,
                "timeout": 10,
                "use_cache": True,
                "cache_duration": 3600
            },
            "security": {
                "encryption_level": "AES-256",
                "hash_algorithm": "SHA-256",
                "session_timeout": 3600,
                "max_login_attempts": 3,
                "enable_firewall": True,
                "enable_intrusion_detection": False
            },
            "performance": {
                "max_memory_gb": 4.0,
                "cpu_threads": 4,
                "gpu_acceleration": False,
                "cache_size_mb": 512,
                "lazy_loading": True,
                "background_loading": True,
                "model_unload_timeout": 60
            },
            "gui": {
                "theme": "dark",
                "font_size": 12,
                "window_width": 1400,
                "window_height": 900,
                "show_fps": False,
                "animations": True
            }
        }

    def _load_user_config(self, config_file: str):
        """Load user configuration"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            self._deep_merge(self.config, user_config)
        except Exception as e:
            print(f"Warning: Failed to load user config: {e}")

    def _deep_merge(self, base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_storage_paths(self) -> Dict[str, Path]:
        """Get all storage paths"""
        return {
            "root": self.storage_path,
            "models": self.storage_path / "models",
            "database": self.storage_path / "database",
            "logs": self.storage_path / "logs",
            "cache": self.storage_path / "cache",
            "backups": self.storage_path / "backups",
            "screenshots": self.storage_path / "screenshots",
            "voice": self.storage_path / "voice_recordings",
            "downloads": self.storage_path / "downloads"
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(self.storage_path)

            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "memory_total_gb": memory.total / (1024 ** 3),
                "memory_available_gb": memory.available / (1024 ** 3),
                "memory_used_percent": memory.percent,
                "disk_total_gb": disk.total / (1024 ** 3),
                "disk_free_gb": disk.free / (1024 ** 3),
                "disk_used_percent": disk.percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
        except Exception as e:
            return {"error": str(e)}

    def check_dangerous_command(self, command: str) -> Dict[str, Any]:
        """Check if a command is dangerous"""
        dangerous_keywords = self.get("automation.blocked_actions", [])
        is_dangerous = any(keyword in command.lower() for keyword in dangerous_keywords)

        return {
            "is_dangerous": is_dangerous,
            "keywords_found": [k for k in dangerous_keywords if k in command.lower()],
            "requires_confirmation": is_dangerous and self.get("automation.dangerous_actions_require_confirmation",
                                                               True)
        }