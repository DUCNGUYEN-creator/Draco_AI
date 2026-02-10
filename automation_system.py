#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Copyright (c) 2026 Nguyen Huu Duc (DUCNGUYEN-creator)
# Project: Draco AI V15 Ultra
#
# This file is part of Draco AI.
# Licensed under the MIT License. See LICENSE file in the project root.
# ------------------------------------------------------------------------------
"""
DRACO AUTOMATION SYSTEM - Complete với lazy loading
"""
from pathlib import Path
import time
import threading
from typing import Dict, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lazy_loader import get_lazy_loader


class DracoAutomation:
    """Automation system với lazy loading"""

    def __init__(self, config):
        self.config = config
        self.initialized = False
        self.safe_mode = config.get("system.safe_mode", False)

        # Lazy loader
        self.lazy_loader = get_lazy_loader()

        # State
        self.permission_level = "normal"
        self.require_confirmation = True

        # Session
        self.session_id = f"auto_{int(time.time())}"

        # Initialize
        self.initialize()

    def initialize(self):
        """Khởi tạo automation (chỉ setup)"""
        if self.initialized:
            return True

        # Check safe mode
        if self.safe_mode:
            print("⚠️ Automation disabled in safe mode")
            return False

        try:
            print("🤖 Initializing Automation System (Lazy Loading)...")

            # Register components
            self._register_components()

            self.initialized = True
            print("✅ Automation System initialized (tools will load on-demand)")
            return True

        except Exception as e:
            print(f"❌ Automation initialization failed: {e}")
            return False

    def _register_components(self):
        """Đăng ký automation components"""

        def load_pyautogui():
            """Load PyAutoGUI on-demand"""
            try:
                import pyautogui

                # Configure safety
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = self.config.get("automation.keyboard_delay", 0.1)

                print("✅ PyAutoGUI loaded")
                return pyautogui
            except ImportError:
                print("❌ pyautogui not installed")
                raise

        # Đăng ký components
        self.lazy_loader.register_component(
            name="pyautogui",
            loader_func=load_pyautogui,
            estimated_memory_mb=10
        )

    def execute_command(self, command_type: str, params: Dict = None) -> Dict[str, Any]:
        """Thực thi command với lazy loading"""
        start_time = time.time()

        if not self.initialized:
            return {"success": False, "error": "Automation not initialized"}

        # Check dangerous commands
        danger_check = self.config.check_dangerous_command(str(params))
        if danger_check["is_dangerous"] and self.require_confirmation:
            return {
                "success": False,
                "error": "Dangerous command requires confirmation",
                "danger_check": danger_check
            }

        try:
            # Load PyAutoGUI on-demand
            pyautogui = self.lazy_loader.get_component("pyautogui")

            result = {"success": True}

            if command_type == "click":
                x = params.get("x", 0)
                y = params.get("y", 0)
                button = params.get("button", "left")

                pyautogui.click(x=x, y=y, button=button)
                result["action"] = f"click at ({x}, {y})"

            elif command_type == "type":
                text = params.get("text", "")
                pyautogui.typewrite(text)
                result["action"] = f"type: {text[:50]}..."

            elif command_type == "press":
                keys = params.get("keys", [])
                pyautogui.hotkey(*keys)
                result["action"] = f"press: {keys}"

            elif command_type == "screenshot":
                filename = params.get("filename", f"screenshot_{int(time.time())}.png")
                screenshot_dir = self.config.get_storage_paths()["screenshots"]
                filename = screenshot_dir / filename

                pyautogui.screenshot(str(filename))
                result["action"] = f"screenshot saved to {filename}"

            else:
                return {"success": False, "error": f"Unknown command: {command_type}"}

            # Schedule unload
            self.lazy_loader.schedule_unload("pyautogui", timeout=30)

            result["processing_time"] = time.time() - start_time
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def get_status(self) -> Dict[str, Any]:
        """Lấy trạng thái automation"""
        loader_status = self.lazy_loader.get_status()

        return {
            "initialized": self.initialized,
            "permission_level": self.permission_level,
            "require_confirmation": self.require_confirmation,
            "session_id": self.session_id,
            "loader_status": loader_status
        }

    def cleanup(self):
        """Dọn dẹp automation"""
        self.lazy_loader.unload_all()
        print("Automation System cleaned up")